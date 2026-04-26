"""Checks for notebook runtime helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from otfm.datasets import sample_8gaussians, sample_moons
from otfm.model import init_mlp_params
from otfm.runtime import make_t_schedule, rollout_euler, rollout_euler_final, train_on_fixed_pairs
from otfm.training import make_optimizer


def test_make_t_schedule_shapes():
    schedule = make_t_schedule(seed=0, steps=5, n=7)
    assert len(schedule) == 5
    assert all(t.shape == (7, 1) for t in schedule)
    assert all(jnp.all((t >= 0.0) & (t <= 1.0)) for t in schedule)


def test_rollout_euler_and_final_match_last_state():
    key = jax.random.PRNGKey(0)
    kp, kx = jax.random.split(key)
    params = init_mlp_params(kp, [3, 16, 16, 2])
    x0 = jax.random.normal(kx, (16, 2))
    traj = rollout_euler(params, x0, steps=12)
    x_final = rollout_euler_final(params, x0, steps=12)
    assert len(traj) == 13
    assert jnp.allclose(traj[-1], x_final, atol=1e-6)


def test_train_on_fixed_pairs_returns_losses():
    key = jax.random.PRNGKey(0)
    k0, k1, kp = jax.random.split(key, 3)
    n = 32
    x0 = sample_moons(k0, n)
    x1 = sample_8gaussians(k1, n)
    params0 = init_mlp_params(kp, [3, 16, 16, 2])
    t_schedule = make_t_schedule(seed=11, steps=4, n=n)
    optimizer = make_optimizer(lr=1e-3)

    params, losses = train_on_fixed_pairs(params0, x0, x1, t_schedule, optimizer=optimizer)

    assert len(losses) == len(t_schedule)
    assert jnp.all(jnp.isfinite(jnp.asarray(losses)))
    assert len(params) == len(params0)
