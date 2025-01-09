import jax.numpy as jnp

# Define the exact solution of the underdamped harmonic oscillator
def oscillator(d, w0, t):
    w = jnp.sqrt(w0 ** 2 - d ** 2)
    phi = jnp.arctan(-d / w)
    A = 1.0 / (2.0 * jnp.cos(phi))
    cos_term = jnp.cos(phi + w * t)
    exp_term = jnp.exp(-d * t)
    x = exp_term * 2 * A * cos_term
    return x
