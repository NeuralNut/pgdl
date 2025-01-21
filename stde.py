import argparse
from collections import namedtuple
from functools import partial
from typing import Callable, NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
from jax.experimental import jet
from jaxtyping import Array, Float
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PINN Training')
parser.add_argument('--SEED', type=int, default=0)
parser.add_argument(
  '--dim', type=int, default=10000
)  # dimension of the problem.
parser.add_argument('--epochs', type=int, default=10000)  # Adam epochs
parser.add_argument('--lr', type=float, default=1e-3)  # Adam lr
parser.add_argument('--PINN_h', type=int, default=128)  # width of PINN
parser.add_argument('--PINN_L', type=int, default=4)  # depth of PINN
parser.add_argument(
  '--N_f', type=int, default=int(100)
)  # num of residual points
parser.add_argument(
  '--N_test', type=int, default=int(20000)
)  # num of test points
parser.add_argument(
  '--test_batch_size', type=int, default=int(200)
)  # num of test points
parser.add_argument('--x_radius', type=float, default=1)
parser.add_argument('--rand_batch_size', type=int, default=10)
parser.add_argument(
  '--sparse',
  action=argparse.BooleanOptionalAction,
  help='whether to use sparse or dense stde'
)
parser.add_argument('--eval_every', type=int, default=1000)
args = parser.parse_args()
print(args)

###########################
# STDE
###########################


def hess_trace(fn: Callable) -> Callable:

  def fn_trace(x_i):
    key = hk.next_rng_key()

    if args.sparse:
      idx_set = jax.random.choice(
        key, args.dim, shape=(args.rand_batch_size,), replace=False
      )
      rand_vec = jax.vmap(lambda i: jnp.eye(args.dim)[i])(idx_set)

    else:
      rand_vec = 2 * (
        jax.random.randint(
          key, shape=(args.rand_batch_size, args.dim), minval=0, maxval=2
        ) - 0.5
      )

    taylor_2 = lambda v: jet.jet(
      fun=fn, primals=(x_i,), series=((v, jnp.zeros(args.dim)),)
    )
    f_vals, (_, hvps) = jax.vmap(taylor_2)(rand_vec)
    trace_est = jnp.mean(hvps)
    if args.sparse:
      trace_est *= args.dim
    return f_vals[0], trace_est

  return fn_trace


###########################
# equation
###########################


def SineGordon_op(x, u) -> Float[Array, "xt_dim"]:
  r"""
  .. math::
  \nabla u(x) + sin(u(x)) = g(x)
  """
  u_, u_xx = hess_trace(u)(x)
  return u_xx + jnp.sin(u_)


coeffs_ = np.random.randn(1, args.dim)


def twobody_sol(x) -> Float[Array, "*batch"]:
  t1 = args.x_radius**2 - jnp.sum(x**2, -1)
  x1, x2 = x[..., :-1], x[..., 1:]
  t2 = coeffs_[:, :-1] * jnp.sin(x1 + jnp.cos(x2) + x2 * jnp.cos(x1))
  t2 = jnp.sum(t2, -1)
  u_exact = jnp.squeeze(t1 * t2)
  return u_exact


def twobody_lapl_analytical(x):
  coeffs = coeffs_[:, :-1]
  const_2 = 1
  u1 = 1 - np.sum(x**2)
  du1_dx = -2 * x
  d2u1_dx2 = -2

  x1, x2 = x[:-1], x[1:]
  coeffs = coeffs.reshape(-1)
  u2 = coeffs * jnp.sin(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1)))
  u2 = jnp.sum(u2)
  du2_dx_part1 = coeffs * jnp.cos(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1))) * \
          const_2 * (1 - x2 * jnp.sin(x1))
  du2_dx_part2 = coeffs * jnp.cos(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1))) * \
          const_2 * (-jnp.sin(x2) + jnp.cos(x1))
  du2_dx = jnp.zeros((args.dim,))
  du2_dx = du2_dx.at[:-1].add(du2_dx_part1)
  du2_dx = du2_dx.at[1:].add(du2_dx_part2)
  d2u2_dx2_part1 = -coeffs * jnp.sin(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1))) * \
      const_2**2 * (1 - x2 * jnp.sin(x1))**2 + \
      coeffs * const_2 * jnp.cos(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1))) * (- x2 * jnp.cos(x1))
  d2u2_dx2_part2 = -coeffs * jnp.sin(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1))) * \
      const_2**2 * (-jnp.sin(x2) + jnp.cos(x1))**2 + \
      coeffs * const_2 * jnp.cos(const_2 * (x1 + jnp.cos(x2) + x2 * jnp.cos(x1))) * \
          (-jnp.cos(x2))
  d2u2_dx2 = jnp.zeros((args.dim,))
  d2u2_dx2 = d2u2_dx2.at[:-1].add(d2u2_dx2_part1)
  d2u2_dx2 = d2u2_dx2.at[1:].add(d2u2_dx2_part2)
  ff = u1 * d2u2_dx2 + 2 * du1_dx * du2_dx + u2 * d2u1_dx2
  ff = jnp.sum(ff)
  u = (u1 * u2)
  return ff, u


@partial(jax.jit, static_argnames=['n_pts'])
def sample_domain_fn(n_pts: int, rng: jax.Array):
  keys = jax.random.split(rng, 6)
  r = jax.random.uniform(keys[0], (n_pts, 1)) * args.x_radius
  x = jax.random.normal(keys[1], (n_pts, args.dim))
  # project x onto the unit sphere, then scale by sampled radius
  x = x / jnp.linalg.norm(x, axis=-1, keepdims=True) * r
  t = jax.random.uniform(keys[2], (n_pts, 1))  # time is 1D
  return x, t, keys[5]


def SineGordon_twobody_inhomo_exact(x):
  u_exact_lapl, u_exact = twobody_lapl_analytical(x)
  g_exact = u_exact_lapl + jnp.sin(u_exact)
  return g_exact


def SineGordon_res_fn(x, u) -> Float[Array, "xt_dim"]:
  r"""
  .. math::
  L u(x) = g(x)
  """
  Lu = SineGordon_op(x, u)
  g = SineGordon_twobody_inhomo_exact(x)
  return Lu - g


def ZeroOnUnitBall_enforce_boundary(x, u_val):
  """Enforece the boundary condition that the u is zero on the unit ball.
  This prevents information leakage from the boundary to the interior."""
  return (args.x_radius**2 - jnp.sum(x**2, -1)) * u_val


###########################
# model
###########################


class PINN(hk.Module):

  def __call__(self, x) -> jax.Array:
    """ansatze for space-time domain scalar function"""
    inputs = x
    pred = jnp.squeeze(self.net(inputs))
    return ZeroOnUnitBall_enforce_boundary(x, pred)

  def net(self, x: jax.Array) -> jax.Array:
    """NOTE: if eqn.time_dependent, the last dim is t."""
    hidden_sizes = [args.PINN_h] * (args.PINN_L - 1) + [1]

    for i, h_i in enumerate(hidden_sizes):
      layer_fn = hk.Linear(h_i, name=f"linear_{i}")
      x = layer_fn(x)
      if i != len(hidden_sizes) - 1:  # activation
        x = jax.nn.tanh(x)
    return x

  def loss_fn(self, x):
    domain_res = jax.vmap(lambda x_: SineGordon_res_fn(x_, self.__call__))(x)
    domain_loss = jnp.mean(domain_res**2)
    loss = domain_loss
    aux = dict(domain_loss=domain_loss)
    return loss, aux

  def err_norms_fn(self, x, y):
    y_pred = self.__call__(x)
    err = y - y_pred
    l1 = jnp.abs(err).sum()
    l2 = (err**2).sum()
    return l1, l2

  def init_for_multitransform(self):
    return (
      self.__call__,
      namedtuple("PINN", ["u", "loss_fn", "err_norms_fn"
                         ])(self.__call__, self.loss_fn, self.err_norms_fn),
    )


###########################
# train
###########################


class TrainingState(NamedTuple):
  params: hk.Params
  opt_state: optax.OptState
  rng_key: jax.Array


def train():
  rng = jax.random.PRNGKey(args.SEED)

  # prepare dummy data for init
  x, _, rng = sample_domain_fn(2, rng)

  # init model
  model = hk.multi_transform(lambda: PINN().init_for_multitransform())
  key, rng = jax.random.split(rng)
  params = model.init(key, x)

  # prepare test data
  n_test_batches = args.N_test // args.test_batch_size
  assert n_test_batches > 0
  x_tests, y_trues = [], []
  for _ in tqdm(range(n_test_batches), desc="generating test data..."):
    # NOTE: why not test boundary condition?
    x_test_i, _, rng = sample_domain_fn(args.test_batch_size, rng)
    y_true_i = twobody_sol(x_test_i)
    x_tests.append(np.array(x_test_i))
    y_trues.append(np.array(y_true_i))

  y_true = jnp.hstack(np.array(y_trues))

  y_true_l1, y_true_l2 = [np.linalg.norm(y_true, ord=ord) for ord in [1, 2]]

  @jax.jit
  def update(state: TrainingState) -> Tuple:
    """sample from domain then update parameter"""
    rng = state.rng_key
    x, _, rng = sample_domain_fn(args.N_f, rng)
    val_and_grads_fn = jax.value_and_grad(model.apply.loss_fn, has_aux=True)
    key, rng = jax.random.split(rng)
    (loss, aux), grad = val_and_grads_fn(state.params, key, x)
    updates, opt_state = optimizer.update(grad, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    return loss, TrainingState(params, opt_state, rng), aux

  # init optimizers
  lr = optax.linear_schedule(
    init_value=args.lr,
    end_value=0,
    transition_steps=args.epochs,
    transition_begin=0
  )
  optimizer = optax.adam(lr)
  opt_state = optimizer.init(params)
  state = TrainingState(params, opt_state, rng)

  err_norms_jit = jax.jit(model.apply.err_norms_fn)

  l1_rel = l2_rel = 0.
  iters = tqdm(range(args.epochs))
  for step in iters:
    loss, state, aux = update(state)

    if step % args.eval_every == 0:  # eval
      l1_total, l2_total_sqr = 0., 0.
      for i in range(n_test_batches):
        l1, l2_sqr = err_norms_jit(
          state.params, state.rng_key, x_tests[i], y_trues[i]
        )
        l1_total += l1
        l2_total_sqr += l2_sqr

      l1_rel = float(l1_total / y_true_l1)
      l2_rel = float(l2_total_sqr**0.5 / y_true_l2)

      desc_str = f"{l1_rel=:.2E} | {l2_rel=:.2E} | {loss=:.2E} | "
      desc_str += " | ".join(
        [f"{k}={v:.2E}" for k, v in aux.items() if v != 0.0]
      )
      print(desc_str)


if __name__ == "__main__":
  train()