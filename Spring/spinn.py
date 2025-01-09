import os
import time
import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from jax import jvp, value_and_grad
from flax import linen as nn
from typing import Sequence
from functools import partial
from flax import linen as nn
from typing import Sequence
from functools import partial

###############################################################################
# 1) SPINN for the damped harmonic oscillator (1D)
###############################################################################
class SPINN_osc(nn.Module):
    """
    SPINN for 1D. We only have one input dimension t.
    `features`: list/tuple specifying widths of each layer.
    Example: features=[64,64,1] => two hidden layers with 64 units each, then output layer with 1 unit.
    """
    features: Sequence[int]

    @nn.compact
    def __call__(self, t):
        """
        t: shape (N,1) or (N,) - we will handle shape carefully
        Returns: shape (N,) or (N,1) for the solution x(t)
        """
        # Just like the original code, we do a feedforward pass:
        # (1) Possibly broadcast t to shape (N,1)
        if t.ndim == 1:
            t = t[:, None]  # shape (N,1)

        init = nn.initializers.glorot_normal()
        X = t
        # hidden layers
        for fs in self.features[:-1]:
            X = nn.Dense(fs, kernel_init=init)(X)
            X = nn.silu(X)
        # final layer
        X = nn.Dense(self.features[-1], kernel_init=init)(X)
        # flatten from (N,1) => (N,)
        return X.squeeze(axis=-1)
    
###############################################################################
# 2) Hessian-Vector Product for second derivative in 1D
###############################################################################
def hvp_fwdfwd_1d(f, primals, tangents, return_primals=False):
    """
    Forward-mode Hessian-vector product in 1D for PDE residuals.
    This is the same pattern as hvp_fwdfwd in the 3D version, but simplified.
    
    f: function R->R, we want d^2 f/dt^2
    primals: (t,)
    tangents: (1,) usually jnp.ones((N,1)) or broadcast
    """
    g = lambda p: jvp(f, (p,), tangents)[1]  # derivative of f w.r.t p
    primals_out, tangents_out = jvp(g, primals, tangents)
    return (primals_out, tangents_out) if return_primals else tangents_out


###############################################################################
# 3) Loss function for the damped oscillator PDE
###############################################################################
def spinn_loss_damped_oscillator(apply_fn, mu, k):
    """
    PDE: x'' + mu x' + k x = 0
    We define a function that returns PDE-residual + initial cond + data losses.
    """

    def _pde_residual(params, t_c):
        """
        PDE residual over collocation points t_c.
        PDE: x_tt + mu x_t + k x = 0
        """
        # x(t)
        u = apply_fn(params, t_c)  # shape (Nc,)
        # We want second derivative in t.
        # We'll do x_tt via hvp. Also x'(t).
        v = jnp.ones_like(t_c)
        
        # x'(t) = derivative of apply_fn wrt t
        # jvp(f, (t_c,), (v,)) gives df/dt.
        # let's define f(t) = apply_fn(params, t)
        def f(t):
            return apply_fn(params, t)

        # x'(t)
        x_t = jvp(f, (t_c,), (v,))[1]

        # x''(t)
        x_tt = hvp_fwdfwd_1d(f, (t_c,), (v,))

        # PDE residual => x_tt + mu x_t + k x
        res = x_tt + mu * x_t + k * u
        return jnp.mean(res**2)  # MSE over collocation points

    def _initial_loss(params, t0):
        """
        For the oscillator, we typically have x(0)=1, x'(0)=0.
        We'll treat t0 as a single float or 0D array for t=0.
        """
        # x(0)
        x0 = apply_fn(params, t0)  # shape (1,) or scalar
        # x'(0)
        v = jnp.ones_like(t0)  # shape (1,)
        def f(t):
            return apply_fn(params, t)
        x0_t = jvp(f, (t0,), (v,))[1]
        # we want (x(0)-1)^2 + (x'(0) - 0)^2
        return jnp.sum((x0 - 1.0)**2 + (x0_t - 0.0)**2)

    def _data_loss(params, t_d, x_d):
        """
        MSE vs. some known data (maybe from exact solution).
        """
        pred = apply_fn(params, t_d)
        return jnp.mean((pred - x_d)**2)

    def loss_fn(params, t_c, t0, t_d, x_d):
        """
        Combine PDE collocation + initial condition + data matching.
        """
        return 1e-4 * _pde_residual(params, t_c) \
             + _initial_loss(params, t0) \
             + _data_loss(params, t_d, x_d)

    return loss_fn


###############################################################################
# 4) Simple data generator for the damped oscillator
###############################################################################
def spinn_train_generator_damped_oscillator(n_colloc, n_data, T, mu, k, key):
    """
    Returns:
      - t_c: collocation points in [0, T]
      - t0: for initial condition
      - t_d, x_d: data points from the exact solution
    """
    # We'll define an "underdamped" oscillator with d>0 => mu=2*d, w0^2 = k, etc.
    # but you can adapt for any damping regime.

    # collocation points
    t_c = jax.random.uniform(key, (n_colloc, 1), minval=0.0, maxval=T).squeeze(-1)  # shape (n_colloc,)
    t_c = jax.lax.stop_gradient(t_c)  # don't backprop through this
    # initial condition at t0=0
    t0 = jnp.array([0.0])  # shape (1,)
    t0 = jax.lax.stop_gradient(t0)  # don't backprop through this

    # data points
    t_d = jnp.linspace(0, T/2, n_data)  # shape (n_data,)
    t_d = jax.lax.stop_gradient(t_d)  # don't backprop through this
    
    # exact solution => x(t) = e^{-d t} * 2A cos(...) or an official formula
    # We'll define a simple function here:
    def exact_solution(t, d, w0):
        # x(t) = e^{-d t} * 2*A*cos(w t + phi)? 
        # For simplicity, let's just do x(0)=1 => 
        # We'll define the same oscillator(d, w0, t) from your prior code:
        w = jnp.sqrt(w0**2 - d**2)
        phi = jnp.arctan(-d / w)
        A = 1.0/(2.0*jnp.cos(phi))
        cos_term = jnp.cos(phi + w*t)
        exp_term = jnp.exp(-d*t)
        return exp_term*2*A*cos_term

    # we have mu=2*d => d=mu/2, w0^2=k => w0=jnp.sqrt(k).
    d = mu / 2
    w0 = jnp.sqrt(k)
    x_d = exact_solution(t_d, d, w0)  # shape (n_data,)

    return t_c, t0, t_d, x_d


###############################################################################
# 5) Putting it all together in main()
###############################################################################
def main_spinn_oscillator(
    n_colloc=100,
    n_data=25,
    T=1.0,
    mu=4.0,
    k=20.0,
    seed=0,
    lr=1e-3,
    epochs=20000,
    n_layers=4,
    width=64,
    log_iter=2000
):
    """
    Example main function demonstrating how to train the SPINN_osc
    on the damped oscillator PDE: x'' + mu x' + k x = 0.
    """
    # fix GPU usage 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    key = jax.random.PRNGKey(seed)
    # network
    feature_sizes = [width]*(n_layers - 1) + [1]  # last layer => 1 output
    model = SPINN_osc(features=feature_sizes)

    # init model
    # We'll initialize with dummy shape (N,1) => let's do N=2
    t_dummy = jnp.ones((2,1))
    params = model.init(key, t_dummy)

    # optimizer
    optim = optax.adam(lr)
    state = optim.init(params)

    # PDE + data
    key, subkey = jax.random.split(key)
    t_c, t0, t_d, x_d = spinn_train_generator_damped_oscillator(n_colloc, n_data, T, mu, k, subkey)
    
    # define PDE loss
    apply_fn = partial(model.apply)
    loss_fn = spinn_loss_damped_oscillator(apply_fn, mu, k)

    @partial(jax.jit, static_argnums=(0,))
    def train_step(loss_f, params, state):
        # single step 
        # collocation points, t0, data, etc. are captured from outer scope
        loss, grads = value_and_grad(loss_f)(params, t_c, t0, t_d, x_d)
        updates, state = optim.update(grads, state)
        params = optax.apply_updates(params, updates)
        return loss, params, state

    # training loop
    losses = []
    tic = time.time()
    for i in trange(1, epochs+1):
        loss_val, params, state = train_step(loss_fn, params, state)
        if i % log_iter == 0:
            losses.append(float(loss_val))
        #     print(f"Epoch {i}/{epochs}, Loss = {loss_val:.5e}")
    toc = time.time()
    print(f"Training took {(toc - tic)/epochs*1000:.2f} ms/iter")

    # final solution + plots
    # Evaluate on a test grid
    t_test = jnp.linspace(0.0, 2*T, 200)
    x_pred = apply_fn(params, t_test)  # shape (200,)

    # Compare with exact solution
    d = mu / 2
    w0 = jnp.sqrt(k)
    def exact_solution(t, d, w0):
        w = jnp.sqrt(w0**2 - d**2)
        phi = jnp.arctan(-d / w)
        A = 1.0/(2.0*jnp.cos(phi))
        cos_term = jnp.cos(phi + w*t)
        exp_term = jnp.exp(-d*t)
        return exp_term*2*A*cos_term

    x_exact = exact_solution(t_test, d, w0)
    # L2 error
    l2_rel = jnp.linalg.norm(x_pred - x_exact)/jnp.linalg.norm(x_exact)
    print(f"Relative L2 error on [0, {T}] ~ {l2_rel:.2e}")

    # quick plot
    plt.figure()
    plt.plot(t_test, x_exact, label='Exact')
    plt.plot(t_test, x_pred, '--', label='SPINN Prediction')
    plt.scatter(t_d, x_d, color='red', label='Data Points')
    plt.title(f"Damped Oscillator: mu={mu}, k={k}")
    plt.xlabel("Time t")
    plt.ylabel("Displacement x(t)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # plot losses
    plt.figure()
    plt.plot(np.arange(len(losses))*log_iter, losses, 'o-')
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("SPINN: Damped Oscillator Training Loss")
    plt.grid(True)
    plt.show()

    return l2_rel


# ###############################################################################
# # 6) Running the script
# ###############################################################################
# if __name__ == "__main__":
#     main_spinn_oscillator(
#         n_colloc=100,
#         n_data=25,
#         T=1.0,     # final time
#         mu=4.0,    # e.g. 2*d => d=2 => mu=4
#         k=400.0,   # e.g. w0^2=400 => w0=20
#         seed=1234,
#         lr=1e-3,
#         epochs=1000,
#         n_layers=4,
#         width=128,
#         log_iter=100
#     )