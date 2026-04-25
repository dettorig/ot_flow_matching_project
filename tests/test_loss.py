"""Sanity checks on the FM loss and the train step."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from otfm.model import init_mlp_params, vf_apply
from otfm.training import fm_loss, make_optimizer, make_train_step


def _setup(n=64, seed=0):
    key = jax.random.PRNGKey(seed)
    k0, k1, kp, kt = jax.random.split(key, 4)
    x0 = jax.random.normal(k0, (n, 2))
    x1 = jax.random.normal(k1, (n, 2)) + 3.0
    t = jax.random.uniform(kt, (n, 1))
    params = init_mlp_params(kp, [3, 32, 32, 2])
    return params, x0, x1, t


def test_fm_loss_is_nonneg_and_finite():
    params, x0, x1, t = _setup()
    loss = fm_loss(params, x0, x1, t)
    assert jnp.isfinite(loss)
    assert float(loss) >= 0.0


def test_fm_loss_zero_when_x0_equals_x1():
    """If x0 == x1, the target velocity is 0 -> any model with vf(x,t)=0 yields zero loss
    in expectation only when the network outputs zero. We instead check the simpler
    invariant: the *target* term contributes 0 since x1 - x0 = 0."""
    params, _, _, t = _setup()
    x = jax.random.normal(jax.random.PRNGKey(1), (64, 2))
    # build a fake (x0, x1) with x0 == x1 == x and compare against pred_v on xt = x
    pred_v = vf_apply(params, x, t)
    expected = jnp.mean(jnp.sum(pred_v**2, axis=1))
    actual = fm_loss(params, x, x, t)
    assert jnp.allclose(actual, expected, atol=1e-5)


def test_train_step_decreases_loss_on_a_batch():
    """One Adam step on a fixed batch should not blow up the loss."""
    params, x0, x1, t = _setup(seed=42)
    optimizer = make_optimizer(lr=1e-2)
    train_step = make_train_step(optimizer)
    opt_state = optimizer.init(params)

    loss0 = float(fm_loss(params, x0, x1, t))
    for _ in range(20):
        params, opt_state, _ = train_step(params, opt_state, x0, x1, t)
    loss1 = float(fm_loss(params, x0, x1, t))

    # After 20 Adam steps on a fixed mini-batch the loss must drop.
    assert loss1 < loss0


def test_fm_loss_grad_is_finite():
    params, x0, x1, t = _setup()
    grad = jax.grad(fm_loss)(params, x0, x1, t)
    flat = jnp.concatenate([jnp.ravel(W) for (W, b) in grad] + [jnp.ravel(b) for (W, b) in grad])
    assert jnp.all(jnp.isfinite(flat))
    assert float(jnp.linalg.norm(flat)) > 0.0
