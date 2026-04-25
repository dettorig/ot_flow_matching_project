"""Multi-seed sweep helpers for fair coupling/epsilon comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import numpy as np
import pandas as pd

from otfm.couplings import (
    hungarian_pairing,
    independent_pairing,
    sample_pairs_from_coupling,
    sinkhorn_barycentric_pairing,
    sinkhorn_coupling,
)
from otfm.datasets import sample_8gaussians, sample_moons
from otfm.metrics import (
    empirical_w2_squared_hungarian,
    normalized_path_energy,
    path_energy,
    trajectory_curvature_metrics,
    velocity_variance_kernel,
)
from otfm.model import init_mlp_params
from otfm.runtime import make_t_schedule, rollout_euler_final, train_on_fixed_pairs


@dataclass
class SweepConfig:
    n: int = 500
    train_steps: int = 2000
    widths: list[int] = field(default_factory=lambda: [3, 128, 128, 128, 2])
    nfe_eval: int = 32
    pe_steps: int = 256
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    eps_list: list[float] = field(default_factory=lambda: [0.01, 0.03, 0.05, 0.1, 0.2])


def run_baseline_sweep(cfg: SweepConfig) -> pd.DataFrame:
    """Run the four canonical couplings (indep / hungarian / sinkhorn sampled / barycentric)
    across multiple seeds and return one row per (seed, coupling).
    """
    rows = []
    for seed in cfg.seeds:
        key = jax.random.PRNGKey(seed)
        key, k0, k1, kpair, kinit, keval = jax.random.split(key, 6)

        x0 = sample_moons(k0, cfg.n)
        x1 = sample_8gaussians(k1, cfg.n)
        base_params = init_mlp_params(kinit, cfg.widths)
        t_schedule = make_t_schedule(seed=1000 + seed, steps=cfg.train_steps, n=cfg.n)

        x0_eval, x1_eval = x0[: cfg.n], x1[: cfg.n]
        w2_data = empirical_w2_squared_hungarian(x0_eval, x1_eval)

        P, _ = sinkhorn_coupling(x0, x1, epsilon=0.05)
        couplings = {
            "independent": independent_pairing(x0, x1),
            "hungarian_exact_ot": hungarian_pairing(x0, x1),
            "sinkhorn_sampled": sample_pairs_from_coupling(kpair, x0, x1, P),
            "sinkhorn_barycentric": sinkhorn_barycentric_pairing(x0, x1, P),
        }

        kv = keval
        for name, (xa, xb) in couplings.items():
            params_tr, losses = train_on_fixed_pairs(base_params, xa, xb, t_schedule)
            pe = path_energy(params_tr, x0_eval, steps=cfg.pe_steps)
            x_gen = rollout_euler_final(params_tr, x0_eval, steps=cfg.nfe_eval)
            w2_endpoint = empirical_w2_squared_hungarian(x_gen, x1_eval)
            kv, kv_sub = jax.random.split(kv)
            target_var = velocity_variance_kernel(
                xa, xb, sigma=0.6, key=kv_sub, include_time=True
            )
            curv = trajectory_curvature_metrics(params_tr, x0_eval, steps=128)

            rows.append(
                {
                    "seed": seed,
                    "coupling": name,
                    "PE": pe,
                    "NPE": normalized_path_energy(pe, w2_data),
                    "endpoint_W2_sq_at_32": float(w2_endpoint),
                    "loss_var": float(np.var(losses)),
                    "target_velocity_kernel_var": target_var,
                    "vel_diff_sq": curv["vel_diff_sq"],
                    "ang_dev": curv["ang_dev"],
                }
            )
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    """Aggregate mean ± std across seeds. Returns flattened-column DataFrame."""
    agg = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in agg.columns.values
    ]
    return agg
