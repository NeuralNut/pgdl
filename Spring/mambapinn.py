################################################################################
#                   MAMBA-PINN (Adapted for continuous inputs)                  #
################################################################################
import numpy as np  # Original NumPy
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit, value_and_grad
from jax.example_libraries import optimizers
from jax.flatten_util import ravel_pytree
from functools import partial
import matplotlib.pyplot as plt
from tqdm import trange
import itertools

# from mambapinn import MambaForPINN
from dataclasses import dataclass
from typing import Union
import math
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.linen.initializers import normal as flax_normal


from data import oscillator

@dataclass
class ModelArgs:
    """
    We will adapt Mamba's arguments to a simple continuous input use-case
    and produce a single output dimension. 
    """
    d_model: int       # hidden dim
    n_layer: int       # number of layers
    d_state: int = 16  # latent state dim
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class RMSNorm(nn.Module):
    d_model: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        weight = self.param('weight', nn.initializers.ones, (self.d_model,))
        normed = x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return normed * weight


class MambaBlock(nn.Module):
    args: ModelArgs

    def setup(self):
        self.in_proj = nn.Dense(
            features=self.args.d_inner * 2,
            kernel_init=flax_normal(),
            use_bias=self.args.bias
        )
        # Depthwise convolution
        self.conv1d = nn.Conv(
            features=self.args.d_inner,
            kernel_size=[self.args.d_conv],
            feature_group_count=self.args.d_inner,
            padding=self.args.d_conv - 1,
            use_bias=self.args.conv_bias
        )
        # x_proj: produces input-specific Δ, B, C
        self.x_proj = nn.Dense(self.args.dt_rank + self.args.d_state * 2, use_bias=False)
        # dt_proj: projects Δ from dt_rank to d_in
        self.dt_proj = nn.Dense(self.args.d_inner, use_bias=True)

        A = jnp.tile(jnp.arange(1, self.args.d_state + 1), (self.args.d_inner, 1))
        self.A_log = self.param('A_log', lambda rng, shape: jnp.log(A), (self.args.d_inner, self.args.d_state))
        self.D = self.param('D', nn.initializers.ones, (self.args.d_inner,))
        self.out_proj = nn.Dense(
            features=self.args.d_model,
            kernel_init=flax_normal(),
            use_bias=self.args.bias
        )

    def __call__(self, x):
        """
        x: (batch, length, d_model)
        Returns: (batch, length, d_model)
        """
        b, l, d = x.shape

        # in_proj => (b, l, 2 * d_inner)
        x_and_res = self.in_proj(x)

        # Split into x, res
        x_part, res = jnp.split(x_and_res, [self.args.d_inner], axis=-1)

        # Depthwise conv => (b, l, d_inner)
        x_part = self.conv1d(x_part)[:, :l, :]

        x_part = jax.nn.silu(x_part) # original Mamba uses SiLU

        # SSM forward pass => (b, l, d_inner)
        y = self.ssm(x_part)

        # Gating => (b, l, d_inner)
        y = y * jax.nn.silu(res) # original Mamba uses SiLU

        # Project back => (b, l, d_model)
        out = self.out_proj(y)
        return out

    def ssm(self, x):
        """
        x: (b, l, d_in) with d_in = d_inner
        Returns: (b, l, d_in)
        """
        d_in, n = self.A_log.shape
        b, l, _ = x.shape

        # A, D input independent
        A = -jnp.exp(self.A_log)  # (d_in, n)
        D = self.D                # (d_in,)

        # x_proj => (b, l, dt_rank + 2*n)
        x_dbl = self.x_proj(x)
        # delta: (b, l, dt_rank), B,C: (b, l, n)
        delta, B, C = jnp.split(x_dbl, [self.args.dt_rank, self.args.dt_rank + n], axis=-1)

        # softplus => positive step sizes
        delta = jax.nn.softplus(self.dt_proj(delta))  # => (b, l, d_in)

        # Selective scan => (b, l, d_in)
        return self.selective_scan(x, delta, A, B, C, D)

    def selective_scan(self, u, delta, A, B, C, D):
        """
        Discretized state space scan across the time dimension.
        u:     (b, l, d_in)
        delta: (b, l, d_in)
        A:     (d_in, n)
        B,C:   (b, l, n)
        D:     (d_in,)
        """
        b, l, d_in = u.shape
        n = A.shape[1]

        # Discretize
        deltaA = jnp.exp(jnp.einsum('bld,dn->bldn', delta, A))
        deltaB_u = jnp.einsum('bld,bln,bld->bldn', delta, B, u)

        x_state = jnp.zeros((b, d_in, n))
        ys = []
        for i in range(l):
            x_state = deltaA[:, i] * x_state + deltaB_u[:, i]
            y_i = jnp.einsum('bdn,bn->bd', x_state, C[:, i, :])

            ys.append(y_i)
        y = jnp.stack(ys, axis=1)  # => (b, l, d_in)

        y = y + u * D
        return y


class ResidualBlock(nn.Module):
    args: ModelArgs

    def setup(self):
        self.mixer = MambaBlock(self.args)
        self.norm = RMSNorm(self.args.d_model)

    def __call__(self, x):
        """
        x: (b, l, d_model)
        """
        return self.mixer(self.norm(x)) + x


class MambaForPINNModel(nn.Module):
    """
    Minimal Mamba that accepts continuous inputs of shape (batch, length, 1)
    and outputs shape (batch, length, 1).

    We'll have multiple residual blocks + final projection to 1D output.
    """
    args: ModelArgs

    def setup(self):
        # A small linear "embedding" from 1 -> d_model
        self.input_proj = nn.Dense(features=self.args.d_model)
        # stack of n_layer Mamba residual blocks
        self.layers = [ResidualBlock(self.args) for _ in range(self.args.n_layer)]
        self.norm_final = RMSNorm(self.args.d_model)
        # final linear from d_model -> 1
        self.output_proj = nn.Dense(features=1)

    def __call__(self, x):
        """
        x: (b, l, 1) -- times
        returns: (b, l, 1)
        """
        # Project up to d_model
        x = self.input_proj(x)
        # Pass through Mamba layers
        for layer in self.layers:
            x = layer(x)
        x = self.norm_final(x)
        # Project down to 1
        y = self.output_proj(x)
        return y


def MambaForPINN(d_model=16, n_layer=2):
    """
    Create `init` and `apply` callables (mimicking the old MLP signature).
    The network will take shape (n,1) and produce (n,1) as outputs,
    using a minimal Mamba-based architecture inside.
    """
    args = ModelArgs(
        d_model=d_model,
        n_layer=n_layer,
        # the rest can remain default
    )
    model = MambaForPINNModel(args)

    def init_fcn(rng_key):
        # We'll do a dummy input shape (1, 1, 1) just to initialize
        dummy_input = jnp.zeros((1, 1, 1))
        return model.init(rng_key, dummy_input)

    def apply_fcn(params, x):
        # x: shape (n,) for time points -> reshape to (n,1,1)
        x_reshaped = x.reshape(-1, 1, 1)
        y = model.apply(params, x_reshaped)  # (n,1,1)
        return y.squeeze(axis=-1)           # => (n,1)

    return init_fcn, apply_fcn



# Define the PINN model
class MAMBAPINN:
    def __init__(self, key, d_model, n_layer, mu, k):

        # ~~~ REPLACEMENT: Use Mamba for PINN instead of MLP ~~~
        self.init, self.apply = MambaForPINN(d_model=d_model, n_layer=n_layer)
        params = self.init(rng_key=key)
        _, self.unravel = ravel_pytree(params)

        # Use optimizers to set optimizer initialization and update functions
        lr = optimizers.exponential_decay(1e-3, decay_steps=5000, decay_rate=0.9999999)
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr)
        self.opt_state = self.opt_init(params)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []

        # Constants for oscillator
        self.mu = mu
        self.k = k

        # Decay rate for time-dependent weighting
        self.lambda_decay = 1.0  # You can adjust this value

    def neural_net(self, params):
        def net_apply(t):
            outputs = self.apply(params, t)  # shape (n,1) if t is shape (n,)
            return outputs.squeeze()         # make sure each scalar is returned
        return net_apply

    def residual_net(self, params):
        net_apply = self.neural_net(params)
        def res_apply(t):
            x = net_apply(t)         # x(t)
            x_t = grad(net_apply)(t) # dx/dt
            x_tt = grad(grad(net_apply))(t) # d^2x/dt^2
            residual = x_tt + self.mu * x_t + self.k * x
            return residual
        return res_apply

    @partial(jit, static_argnums=(0,))
    def loss(self, params, t_r, t_data, x_data):
        # Residual loss with time-dependent weighting
        res_apply = self.residual_net(params)
        residual_fn = vmap(res_apply)
        res = residual_fn(t_r)
        res_squared = res ** 2
        loss_res = jnp.sum(res_squared)

        # # Time-dependent weights
        # weights_time = jnp.exp(-self.lambda_decay * t_r)
        # weights_time /= jnp.sum(weights_time)  # Normalize weights
        # weights = weights_time
        # weights /= jnp.sum(weights)

        # loss_res = jnp.sum(weights * res_squared)

        # Data loss (at observed data points)
        net_apply = self.neural_net(params)
        x_pred = vmap(net_apply)(t_data)
        loss_data = jnp.mean((x_pred - x_data) ** 2)

        # Initial condition loss
        x0_pred = net_apply(jnp.array(0.0))
        x0_t_pred = grad(net_apply)(jnp.array(0.0))
        loss_ic = (x0_pred - 1.0) ** 2 + (x0_t_pred - 0.0) ** 2

        # Total loss
        total_loss = 1.e-4 * loss_res + loss_data + loss_ic

        return total_loss

    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, t_r, t_data, x_data):
        params = self.get_params(opt_state)
        value, grads = value_and_grad(self.loss)(params, t_r, t_data, x_data)
        opt_state = self.opt_update(i, grads, opt_state)
        return opt_state, value

    def train(self, nIter, t_r, t_data, x_data):
        pbar = trange(nIter)
        
        # Main training loop
        for it in pbar:
            self.current_count = next(self.itercount)
            self.opt_state, loss_value = self.step(
                self.current_count, self.opt_state, t_r, t_data, x_data
            )
            if it % 100 == 0:
                self.loss_log.append(loss_value)
                pbar.set_postfix({'Loss': loss_value})


d = 2.0
w0 = 20.0
mu = 2*d
k = w0**2

# Generate collocation points and data points
t_r = jnp.linspace(0.0, 1.0, 100)   # Collocation points for residual
t_data = jnp.linspace(0.0, 0.5, 25) # Data points for initial training
x_data = oscillator(d, w0, t_data)  # Exact solution at data points

# Initialize the model
key = random.PRNGKey(0)
model = MAMBAPINN(key, d_model=128, n_layer=4, mu=mu, k=k)

# Convert data to appropriate types
t_r = jnp.array(t_r)
t_data = jnp.array(t_data)
x_data = jnp.array(x_data)

# Adjust the decay rate as needed
model.lambda_decay = 1.0  # You can experiment with different values

# Train the model
model.train(nIter=1000, t_r=t_r, t_data=t_data, x_data=x_data)

# Evaluate the model
params = model.get_params(model.opt_state)
t_test = jnp.linspace(0.0, 2.0, 200)
t_test = jnp.array(t_test)
net_apply = model.neural_net(params)
x_pred = vmap(net_apply)(t_test)
x_exact = oscillator(d, w0, t_test)

# Plot the results
plt.figure()
plt.plot(t_test, x_exact, label='Exact Solution')
plt.plot(t_test, x_pred, '--', label='PINN Prediction (Mamba-based)')
plt.scatter(t_data, x_data, color='red', label='Training Data')
plt.legend()
plt.xlabel('Time $t$')
plt.ylabel('Displacement $x(t)$')
plt.title('Under-damped Harmonic Oscillator (Mamba-based PINN)')
plt.show()

# Plot the training loss
plt.figure()
plt.plot(np.arange(len(model.loss_log))*100, model.loss_log)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Training Loss (Mamba-based PINN)')
plt.show()
