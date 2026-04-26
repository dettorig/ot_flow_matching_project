"""Flow Matching loss and JIT-compiled training step."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from otfm.model import vf_apply


def fm_loss(params, x0, x1, t):
    """Conditional Flow Matching loss with linear interpolation path.

    L = E_{x0,x1,t} || v_theta((1-t)x0 + t x1, t) - (x1 - x0) ||^2
    """
    xt = (1.0 - t) * x0 + t * x1
    target_v = x1 - x0
    pred_v = vf_apply(params, xt, t)
    return jnp.mean(jnp.sum((pred_v - target_v) ** 2, axis=1))


def make_optimizer(lr: float = 1e-3):
    return optax.adam(lr)


def make_train_step(optimizer):
    """Return a JIT-compiled train_step closed over the optimizer."""

    @jax.jit
    def train_step(params, opt_state, x0, x1, t):
        loss, grads = jax.value_and_grad(fm_loss)(params, x0, x1, t)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step
