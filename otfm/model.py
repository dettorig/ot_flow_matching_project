"""Tiny MLP velocity field v_theta(x, t) implemented in raw JAX."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from typing import Sequence

Array = jnp.ndarray
LayerParams = tuple[Array, Array]  # (W, b)
MLPParams = list[LayerParams]


def init_mlp_params(key: Array, widths: Sequence[int], w_scale: float = 0.1) -> MLPParams:
    """Initialise an MLP with layer widths [din, ..., dout]."""
    if len(widths) < 2:
        raise ValueError("widths must contain at least input and output dimensions.")

    params: MLPParams = []
    keys = jax.random.split(key, len(widths) - 1)

    for k, (din, dout) in zip(keys, zip(widths[:-1], widths[1:], strict=True), strict=True):
        w_key, _ = jax.random.split(k)
        W = w_scale * jax.random.normal(w_key, (din, dout))
        b = jnp.zeros((dout,))
        params.append((W, b))

    return params


def mlp_apply(params: MLPParams, x: Array) -> Array:
    """Apply MLP to x with SiLU on hidden layers and linear output."""
    h = x
    for i, (W, b) in enumerate(params):
        h = h @ W + b
        if i < len(params) - 1:
            h = jax.nn.silu(h)
    return h


def vf_apply(params: MLPParams, x: Array, t: Array) -> Array:
    """Velocity field. x: (B,d), t: (B,1), returns (B,d)."""
    inp = jnp.concatenate([x, t], axis=1)
    return mlp_apply(params, inp)
