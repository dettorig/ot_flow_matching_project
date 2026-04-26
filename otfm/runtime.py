"""Runtime helpers: schedules, fixed-pair training loop, and Euler rollout utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from otfm.model import vf_apply
from otfm.training import make_optimizer, make_train_step


def make_t_schedule(seed: int, steps: int, n: int):
    """Return a list of (n,1) time samples reused across couplings for fairness."""
    k = jax.random.PRNGKey(seed)
    keys = jax.random.split(k, steps)
    return [jax.random.uniform(ki, (n, 1)) for ki in keys]


def rollout_euler(params, x_init, steps: int = 32):
    """Euler integration on [0,1]. Returns full trajectory list (length steps+1)."""
    dt = 1.0 / steps
    x = x_init
    traj = [x]
    for k in range(steps):
        tval = jnp.full((x.shape[0], 1), k / steps)
        v = vf_apply(params, x, tval)
        x = x + dt * v
        traj.append(x)
    return traj


def rollout_euler_final(params, x_init, steps: int):
    """Euler integration on [0,1]. Returns only final state."""
    dt = 1.0 / steps
    x = x_init
    for k in range(steps):
        tval = jnp.full((x.shape[0], 1), k / steps)
        v = vf_apply(params, x, tval)
        x = x + dt * v
    return x


def train_on_fixed_pairs(params0, x0_pair, x1_pair, t_schedule, optimizer=None):
    """Train FM on a fixed pairing (x0_pair, x1_pair) over a precomputed t_schedule."""
    optimizer = optimizer or make_optimizer()
    train_step = make_train_step(optimizer)

    params = params0
    opt_state = optimizer.init(params)
    losses = []

    for t in t_schedule:
        params, opt_state, loss = train_step(params, opt_state, x0_pair, x1_pair, t)
        losses.append(float(loss))

    return params, np.asarray(losses)
