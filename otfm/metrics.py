"""Quantitative metrics for trained Flow Matching models and couplings."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

from otfm.model import vf_apply


# ---------- transport-cost based ----------

def empirical_w2_squared_hungarian(A, B) -> float:
    """Average squared matching cost between equal-size samples."""
    C = jnp.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
    row_ind, col_ind = linear_sum_assignment(np.asarray(C))
    return float(np.asarray(C)[row_ind, col_ind].mean())


def sliced_wasserstein_2(A, B, n_projections: int = 256, key=None) -> float:
    """Sliced Wasserstein-2 (squared) between equal-size samples."""
    if key is None:
        key = jax.random.PRNGKey(0)

    A = jnp.asarray(A)
    B = jnp.asarray(B)
    d = A.shape[1]

    theta = jax.random.normal(key, (n_projections, d))
    theta = theta / (jnp.linalg.norm(theta, axis=1, keepdims=True) + 1e-12)

    proj_a = A @ theta.T
    proj_b = B @ theta.T

    sa = jnp.sort(proj_a, axis=0)
    sb = jnp.sort(proj_b, axis=0)
    sw2 = jnp.mean((sa - sb) ** 2)
    return float(sw2)


# ---------- path-based ----------

def path_energy(params, x_init, steps: int = 256) -> float:
    """PE = E[∫_0^1 ||v_theta(x_t, t)||^2 dt] estimated by Euler quadrature."""
    dt = 1.0 / steps
    x = x_init
    acc = 0.0
    for k in range(steps):
        tval = jnp.full((x.shape[0], 1), k / steps)
        v = vf_apply(params, x, tval)
        acc += float(jnp.mean(jnp.sum(v * v, axis=1))) * dt
        x = x + dt * v
    return acc


def normalized_path_energy(pe: float, w2_data: float) -> float:
    """NPE = |PE - W2_data| / W2_data."""
    return abs(pe - w2_data) / (w2_data + 1e-12)


def trajectory_curvature_metrics(params, x_init, steps: int = 128, eps: float = 1e-8):
    """Curvature-style metrics on Euler trajectories."""
    dt = 1.0 / steps
    x = x_init
    vel_diffs = []
    ang_devs = []

    t0 = jnp.zeros((x.shape[0], 1))
    v_prev = vf_apply(params, x, t0)

    for k in range(1, steps + 1):
        x = x + dt * v_prev
        tval = jnp.full((x.shape[0], 1), k / steps)
        v_curr = vf_apply(params, x, tval)

        dv2 = jnp.sum((v_curr - v_prev) ** 2, axis=1)
        vel_diffs.append(jnp.mean(dv2))

        num = jnp.sum(v_curr * v_prev, axis=1)
        den = jnp.linalg.norm(v_curr, axis=1) * jnp.linalg.norm(v_prev, axis=1) + eps
        cos = jnp.clip(num / den, -1.0, 1.0)
        ang_devs.append(jnp.mean(1.0 - cos))

        v_prev = v_curr

    return {
        "vel_diff_sq": float(jnp.mean(jnp.stack(vel_diffs))),
        "ang_dev": float(jnp.mean(jnp.stack(ang_devs))),
    }


# ---------- training proxies ----------

def velocity_variance_kernel(
    xa,
    xb,
    n_time_samples: int = 8,
    sigma: float = 0.5,
    key=None,
    n_query: int = 1024,
    n_ref: int = 2048,
    include_time: bool = False,
    time_scale: float = 1.0,
) -> float:
    """Estimate E[Var(v | x_t)] using kernel-conditional moments."""
    if key is None:
        key = jax.random.PRNGKey(0)

    xa = jnp.asarray(xa)
    xb = jnp.asarray(xb)

    N, d = xa.shape
    vel0 = xb - xa

    key_t, key_q, key_r = jax.random.split(key, 3)
    t = jax.random.uniform(key_t, (n_time_samples, N, 1))

    xt = (1.0 - t) * xa[None, :, :] + t * xb[None, :, :]
    vel = jnp.broadcast_to(vel0[None, :, :], (n_time_samples, N, d))

    xt = xt.reshape(-1, d)
    vel = vel.reshape(-1, d)
    t_flat = t.reshape(-1, 1)
    M = xt.shape[0]

    feat = jnp.concatenate([xt, time_scale * t_flat], axis=1) if include_time else xt

    nq = int(min(n_query, M))
    nr = int(min(n_ref, M))

    q_idx = jax.random.choice(key_q, M, shape=(nq,), replace=False)
    r_idx = jax.random.choice(key_r, M, shape=(nr,), replace=False)

    fq = feat[q_idx]
    fr = feat[r_idx]
    v_ref = vel[r_idx]

    diff = fq[:, None, :] - fr[None, :, :]
    dist2 = jnp.sum(diff**2, axis=-1)

    K = jnp.exp(-dist2 / (2.0 * sigma**2))
    K = K / (jnp.sum(K, axis=1, keepdims=True) + 1e-8)

    mean_v = K @ v_ref
    mean_v2 = K @ (v_ref**2)

    var = jnp.maximum(mean_v2 - mean_v**2, 0.0)
    local_var = jnp.sum(var, axis=1)

    return float(jnp.mean(local_var))


# ---------- mode coverage / geometry ----------

def nearest_center_idx(x, centers):
    """Return nearest-center index for each sample."""
    d2 = jnp.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    return jnp.argmin(d2, axis=1)


def occupancy_hist(x, centers, n_modes: int = 8):
    """Histogram + probabilities of nearest-mode occupancy."""
    idx = nearest_center_idx(x, centers)
    counts = np.bincount(np.asarray(idx), minlength=n_modes).astype(np.float64)
    probs = counts / max(counts.sum(), 1.0)
    return counts, probs


def mode_distance_metrics(x, centers):
    """Mean and q90 squared distance to nearest center."""
    d2 = jnp.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    min_d2 = jnp.min(d2, axis=1)
    return float(jnp.mean(min_d2)), float(jnp.quantile(min_d2, 0.9))


def occupancy_kl(target_probs, gen_probs) -> float:
    """KL(target || generated) on occupancy distributions."""
    target_probs = np.asarray(target_probs)
    gen_probs = np.asarray(gen_probs)
    return float(
        np.sum(target_probs * (np.log(target_probs + 1e-12) - np.log(gen_probs + 1e-12)))
    )
