"""Tiny MLP velocity field v_theta(x, t) implemented in raw JAX (no Flax/Haiku)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def init_mlp_params(key, widths):
    """Initialise an MLP with the given layer widths (input first, output last)."""
    params = []
    keys = jax.random.split(key, len(widths) - 1)
    for k, (din, dout) in zip(keys, zip(widths[:-1], widths[1:], strict=True), strict=True):
        w_key, _ = jax.random.split(k)
        W = 0.1 * jax.random.normal(w_key, (din, dout))
        b = jnp.zeros((dout,))
        params.append((W, b))
    return params


def mlp_apply(params, x):
    h = x
    for i, (W, b) in enumerate(params):
        h = h @ W + b
        if i < len(params) - 1:
            h = jax.nn.silu(h)
    return h


def vf_apply(params, x, t):
    """Velocity field. x: (B, d), t: (B, 1). Returns (B, d)."""
    inp = jnp.concatenate([x, t], axis=1)
    return mlp_apply(params, inp)
