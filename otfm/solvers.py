"""ODE solvers for FM rollout.

All solvers integrate dx/dt = v_theta(x, t) on [0, 1] with a fixed step size
1/steps, starting from x_init at t=0. The ``return_traj`` flag controls
whether the function returns the full list of intermediate states (for plots
and curvature metrics) or only the final state (for endpoint metrics).
"""

from __future__ import annotations

import jax.numpy as jnp

from otfm.model import vf_apply


def _t_col(x, t):
    return jnp.full((x.shape[0], 1), t)


def euler(params, x_init, steps: int, return_traj: bool = False):
    """Forward Euler. NFE = steps."""
    dt = 1.0 / steps
    x = x_init
    traj = [x]
    for k in range(steps):
        v = vf_apply(params, x, _t_col(x, k / steps))
        x = x + dt * v
        traj.append(x)
    return traj if return_traj else x


def heun(params, x_init, steps: int, return_traj: bool = False):
    """Heun's method (improved Euler / RK2). NFE = 2 * steps."""
    dt = 1.0 / steps
    x = x_init
    traj = [x]
    for k in range(steps):
        t0 = k / steps
        t1 = (k + 1) / steps
        k1 = vf_apply(params, x, _t_col(x, t0))
        x_pred = x + dt * k1
        k2 = vf_apply(params, x_pred, _t_col(x_pred, t1))
        x = x + 0.5 * dt * (k1 + k2)
        traj.append(x)
    return traj if return_traj else x


def rk4(params, x_init, steps: int, return_traj: bool = False):
    """Classical Runge-Kutta of order 4. NFE = 4 * steps."""
    dt = 1.0 / steps
    x = x_init
    traj = [x]
    for k in range(steps):
        t0 = k / steps
        tm = (k + 0.5) / steps
        t1 = (k + 1) / steps
        k1 = vf_apply(params, x, _t_col(x, t0))
        k2 = vf_apply(params, x + 0.5 * dt * k1, _t_col(x, tm))
        k3 = vf_apply(params, x + 0.5 * dt * k2, _t_col(x, tm))
        k4 = vf_apply(params, x + dt * k3, _t_col(x, t1))
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj.append(x)
    return traj if return_traj else x


SOLVERS = {"euler": euler, "heun": heun, "rk4": rk4}


def get_solver(name: str):
    if name not in SOLVERS:
        raise ValueError(f"unknown solver {name!r}; choose from {sorted(SOLVERS)}")
    return SOLVERS[name]


def nfe_per_step(name: str) -> int:
    """Number of velocity-field evaluations per step for each solver."""
    return {"euler": 1, "heun": 2, "rk4": 4}[name]
