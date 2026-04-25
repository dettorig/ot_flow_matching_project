"""Quantitative metrics for trained Flow Matching models and couplings.

Includes:
- empirical W2^2 between two equal-size samples (Hungarian estimate),
- Path Energy and NPE (normalised against the ground-truth W2^2 of the data),
- training proxies via kernel-conditional velocity variance,
- trajectory curvature / angular deviation,
- mode coverage / mode-distance metrics for the 8-Gaussians target,
- Sliced Wasserstein-2 (random projections, closed-form 1D OT).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

from otfm.model import vf_apply

# ---------- transport-cost based ----------


def empirical_w2_squared_hungarian(A, B) -> float:
    """Average squared matching cost between two equal-size samples.

    Used as an estimator of W_2^2 between empirical distributions of A and B.
    """
    C = jnp.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
    row_ind, col_ind = linear_sum_assignment(np.asarray(C))
    return float(np.asarray(C)[row_ind, col_ind].mean())


def sliced_wasserstein_2(A, B, n_projections: int = 256, key=None) -> float:
    """Sliced Wasserstein-2 between equal-size samples A, B.

    SW_2^2(A, B) = E_{theta} W_2^2(theta^T A, theta^T B), estimated by Monte Carlo
    over directions and closed-form 1D OT (sort and pair by quantile).
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    A = jnp.asarray(A)
    B = jnp.asarray(B)
    d = A.shape[1]
    theta = jax.random.normal(key, (n_projections, d))
    theta = theta / (jnp.linalg.norm(theta, axis=1, keepdims=True) + 1e-12)
    proj_a = A @ theta.T  # (nA, P)
    proj_b = B @ theta.T  # (nB, P)
    sa = jnp.sort(proj_a, axis=0)
    sb = jnp.sort(proj_b, axis=0)
    sw = jnp.mean((sa - sb) ** 2)
    return float(sw)


# ---------- path-based ----------


def path_energy(params, x_init, steps: int = 256) -> float:
    """PE = E[ \\int_0^1 ||v_theta(x_t, t)||^2 dt ] estimated by Euler quadrature."""
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
    """NPE = |PE - W2^2(data)| / W2^2(data) — relative excess transport cost along trajectories."""
    return abs(pe - w2_data) / (w2_data + 1e-12)


def trajectory_curvature_metrics(params, x_init, steps: int = 128, eps: float = 1e-8):
    """Two curvature-style metrics on Euler trajectories.

    Returns:
      vel_diff_sq : E[ ||v_{t+dt} - v_t||^2 ]
      ang_dev     : E[ 1 - cos(v_{t+dt}, v_t) ]   (0 = perfectly aligned trajectory)
    """
    dt = 1.0 / steps
    x = x_init
    vel_diffs, ang_devs = [], []

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
    """Estimate E[ Var(v | x_t) ] using kernel-conditional moments.

    Works on the (xa, xb) pairs alone (no trained model needed) and gives a
    cheap proxy for the irreducible noise the FM regression has to fit.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    N, d = xa.shape
    vel0 = xb - xa  # (N, d)

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


# ---------- mode coverage (8-Gaussians target) ----------


def nearest_center_idx(x, centers):
    d2 = jnp.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    return jnp.argmin(d2, axis=1)


def occupancy_hist(x, centers, n_modes: int = 8):
    idx = nearest_center_idx(x, centers)
    counts = np.bincount(np.asarray(idx), minlength=n_modes).astype(np.float64)
    probs = counts / max(counts.sum(), 1.0)
    return counts, probs


def mode_distance_metrics(x, centers):
    d2 = jnp.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    min_d2 = jnp.min(d2, axis=1)
    return float(jnp.mean(min_d2)), float(jnp.quantile(min_d2, 0.9))


def occupancy_kl(target_probs, gen_probs) -> float:
    """KL(target || gen) on mode occupancy histograms (with smoothing)."""
    target_probs = np.asarray(target_probs)
    gen_probs = np.asarray(gen_probs)
    return float(
        np.sum(target_probs * (np.log(target_probs + 1e-12) - np.log(gen_probs + 1e-12)))
    )
