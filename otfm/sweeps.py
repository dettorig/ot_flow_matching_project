"""Multi-seed sweep helpers for coupling/epsilon comparisons."""

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
    n: int = 1000
    train_steps: int = 2000
    widths: list[int] = field(default_factory=lambda: [3, 128, 128, 128, 2])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    eps: float = 0.05
    nfe_list: list[int] = field(default_factory=lambda: [4, 8, 16, 32, 64])
    pe_steps: int = 256
    curvature_steps: int = 128
    t_schedule_seed_offset: int = 1000
    velocity_kernel_sigma: float = 0.6
    velocity_kernel_include_time: bool = True


def run_baseline_sweep(cfg: SweepConfig, return_low_nfe: bool = False):
    """Run baseline 4 couplings across seeds.

    Returns:
      - summary_df by default (backward compatible),
      - (summary_df, low_nfe_df) if return_low_nfe=True.
    """
    summary_rows = []
    low_nfe_rows = []

    for seed in cfg.seeds:
        key = jax.random.PRNGKey(seed)
        key, k0, k1, kpair, kinit, keval = jax.random.split(key, 6)

        # Same baseline data protocol as firstrun
        x0 = sample_moons(k0, cfg.n)
        x1 = sample_8gaussians(k1, cfg.n)

        base_params = init_mlp_params(kinit, cfg.widths)
        t_schedule = make_t_schedule(
            seed=cfg.t_schedule_seed_offset + seed,
            steps=cfg.train_steps,
            n=cfg.n,
        )

        x0_eval, x1_eval = x0, x1
        w2_data = empirical_w2_squared_hungarian(x0_eval, x1_eval)

        P, _ = sinkhorn_coupling(x0, x1, epsilon=cfg.eps)
        coupling_name_sink = f"sinkhorn_eps_{cfg.eps}"

        couplings = {
            "independent": independent_pairing(x0, x1),
            "hungarian_exact_ot": hungarian_pairing(x0, x1),
            coupling_name_sink: sample_pairs_from_coupling(kpair, x0, x1, P),
            "sinkhorn_barycentric": sinkhorn_barycentric_pairing(x0, x1, P),
        }

        kv = keval
        for name, (xa, xb) in couplings.items():
            params_tr, losses = train_on_fixed_pairs(base_params, xa, xb, t_schedule)

            pe = path_energy(params_tr, x0_eval, steps=cfg.pe_steps)
            npe = normalized_path_energy(pe, w2_data)

            # endpoint at 32 for direct firstrun comparability
            x_gen_32 = rollout_euler_final(params_tr, x0_eval, steps=32)
            endpoint_w2_sq_32 = empirical_w2_squared_hungarian(x_gen_32, x1_eval)

            # low-NFE table
            for nfe in cfg.nfe_list:
                x_gen = rollout_euler_final(params_tr, x0_eval, steps=nfe)
                w2_ep = empirical_w2_squared_hungarian(x_gen, x1_eval)
                low_nfe_rows.append(
                    {
                        "seed": seed,
                        "coupling": name,
                        "nfe": nfe,
                        "endpoint_w2_sq": float(w2_ep),
                    }
                )

            kv, kv_sub = jax.random.split(kv)
            target_var = velocity_variance_kernel(
                xa,
                xb,
                sigma=cfg.velocity_kernel_sigma,
                key=kv_sub,
                include_time=cfg.velocity_kernel_include_time,
            )

            curv = trajectory_curvature_metrics(params_tr, x0_eval, steps=cfg.curvature_steps)

            summary_rows.append(
                {
                    "seed": seed,
                    "coupling": name,
                    "PE": float(pe),
                    "NPE": float(npe),
                    "W2_data_sq": float(w2_data),
                    "endpoint_W2_sq_at_32": float(endpoint_w2_sq_32),
                    "loss_mean": float(np.mean(losses)),
                    "loss_var": float(np.var(losses)),
                    "target_velocity_kernel_var": float(target_var),
                    "vel_diff_sq": float(curv["vel_diff_sq"]),
                    "ang_dev": float(curv["ang_dev"]),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    low_nfe_df = pd.DataFrame(low_nfe_rows)

    if return_low_nfe:
        return summary_df, low_nfe_df
    return summary_df


def aggregate(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    """Aggregate mean ± std across seeds."""
    agg = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in agg.columns.values
    ]
    return agg
