"""Notebook diagnostics"""

from __future__ import annotations

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from otfm.model import vf_apply


def learned_vs_target_metrics(
    params,
    xa,
    xb,
    key,
    holdout_frac=0.2,
    n_time_samples=8,
    max_train_pairs=1024,
    max_eval_pairs=1024,
):
    xa = jnp.asarray(xa)
    xb = jnp.asarray(xb)
    n = xa.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 pairs.")

    k_perm, k_train_sub, k_eval_sub, k_t_train, k_t_eval = jax.random.split(key, 5)

    perm = jax.random.permutation(k_perm, n)
    n_eval = max(1, int(holdout_frac * n))
    n_eval = min(n_eval, n - 1)
    n_train = n - n_eval

    train_idx = perm[:n_train]
    eval_idx = perm[n_train:]

    def _subsample(idxs, k_sub, max_pairs):
        m = idxs.shape[0]
        if m <= max_pairs:
            return idxs
        return jax.random.choice(k_sub, idxs, shape=(max_pairs,), replace=False)

    train_idx = _subsample(train_idx, k_train_sub, max_train_pairs)
    eval_idx = _subsample(eval_idx, k_eval_sub, max_eval_pairs)

    xa_train, xb_train = xa[train_idx], xb[train_idx]
    xa_eval, xb_eval = xa[eval_idx], xb[eval_idx]

    def _estimate(xa_part, xb_part, k_t):
        m = xa_part.shape[0]
        t = jax.random.uniform(k_t, (n_time_samples, m, 1))

        xt = (1.0 - t) * xa_part[None, :, :] + t * xb_part[None, :, :]
        target_single = xb_part - xa_part
        target = jnp.broadcast_to(target_single[None, :, :], xt.shape)

        xt_flat = xt.reshape(-1, xa_part.shape[1])
        t_flat = t.reshape(-1, 1)
        target_flat = target.reshape(-1, xb_part.shape[1])

        pred_flat = vf_apply(params, xt_flat, t_flat)

        sq_err = jnp.sum((pred_flat - target_flat) ** 2, axis=1)
        mse = jnp.mean(sq_err)

        target_norm = jnp.mean(jnp.sum(target_flat ** 2, axis=1))
        nmse = mse / (target_norm + 1e-8)

        pred_norm = jnp.linalg.norm(pred_flat, axis=1)
        target_norm_vec = jnp.linalg.norm(target_flat, axis=1)
        cos = jnp.sum(pred_flat * target_flat, axis=1) / (pred_norm * target_norm_vec + 1e-8)

        return float(mse), float(nmse), float(jnp.mean(cos))

    target_mse_train, target_nmse_train, target_cosine_train = _estimate(xa_train, xb_train, k_t_train)
    target_mse_eval, target_nmse_eval, target_cosine_eval = _estimate(xa_eval, xb_eval, k_t_eval)

    return {
        "target_mse_train": target_mse_train,
        "target_mse_eval": target_mse_eval,
        "target_nmse_train": target_nmse_train,
        "target_nmse_eval": target_nmse_eval,
        "target_cosine_train": target_cosine_train,
        "target_cosine_eval": target_cosine_eval,
    }


def vf_single_jm(params, x, t_scalar):
    return vf_apply(params, x[None, :], jnp.array([[t_scalar]]))[0]


def jacobian_x_single_jm(params, x, t_scalar):
    return jax.jacfwd(lambda xx: vf_single_jm(params, xx, t_scalar))(x)


def dt_single_jm(params, x, t_scalar):
    return jax.jacfwd(lambda tt: vf_single_jm(params, x, tt))(t_scalar)


def material_acc_single_jm(params, x, t_scalar):
    v = vf_single_jm(params, x, t_scalar)
    J = jacobian_x_single_jm(params, x, t_scalar)
    vt = dt_single_jm(params, x, t_scalar)
    return vt + J @ v


vmap_jac_jm = jax.vmap(lambda x, p, t: jacobian_x_single_jm(p, x, t), in_axes=(0, None, None))
vmap_acc_jm = jax.vmap(lambda x, p, t: material_acc_single_jm(p, x, t), in_axes=(0, None, None))


def jacobian_material_metrics_on_traj(
    params,
    traj_list,
    times,
    max_points=256,
    time_stride=4,
    rng_seed=0,
):
    rng = np.random.default_rng(rng_seed)
    rows = []

    T = len(traj_list)
    assert len(times) == T, "times and trajectory length mismatch"

    time_idx = np.arange(0, T, time_stride, dtype=int)
    if time_idx[-1] != T - 1:
        time_idx = np.append(time_idx, T - 1)

    for k in time_idx:
        tval = float(times[k])
        xk = np.asarray(traj_list[k])

        n = xk.shape[0]
        if n > max_points:
            idx = rng.choice(n, size=max_points, replace=False)
            xk = xk[idx]

        xk_j = jnp.asarray(xk)

        J_all = vmap_jac_jm(xk_j, params, tval)
        a_all = vmap_acc_jm(xk_j, params, tval)

        jac_fro = jnp.linalg.norm(J_all, axis=(1, 2))
        acc_norm = jnp.linalg.norm(a_all, axis=1)

        jac_np = np.asarray(jac_fro)
        acc_np = np.asarray(acc_norm)

        rows.append(
            {
                "t": tval,
                "jacobian_fro_mean": float(jac_np.mean()),
                "jacobian_fro_p95": float(np.percentile(jac_np, 95)),
                "acceleration_mean": float(acc_np.mean()),
                "acceleration_p95": float(np.percentile(acc_np, 95)),
            }
        )

    return pd.DataFrame(rows).sort_values("t").reset_index(drop=True)


def target_conflict_score_cfh(xa, xb, t_value=0.5, k=32, max_points=1200, seed=0):
    xa = np.asarray(xa)
    xb = np.asarray(xb)

    n = xa.shape[0]
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        xa = xa[idx]
        xb = xb[idx]
        n = max_points

    xt = (1.0 - t_value) * xa + t_value * xb
    u = xb - xa
    u_norm = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-8)

    d2 = np.sum((xt[:, None, :] - xt[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)

    k_eff = int(min(k, n - 1))
    if k_eff <= 0:
        return np.nan

    nn_idx = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]

    neigh_u = u_norm[nn_idx]
    self_u = u_norm[:, None, :]
    cos = np.sum(neigh_u * self_u, axis=-1)

    conflict = 1.0 - cos
    return float(np.mean(conflict))
