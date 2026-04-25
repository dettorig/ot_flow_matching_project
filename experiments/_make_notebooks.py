"""Tiny helper to (re)generate the experiment notebook skeletons.

We keep the notebooks under version control as compact JSON without outputs
(see ``nbstripout``). This script regenerates them from a single Python source
of truth — running ``python experiments/_make_notebooks.py`` after editing the
strings below produces fresh ipynb files.
"""

from __future__ import annotations

import json
import pathlib
from typing import Sequence


def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md_cell(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def write_nb(path: pathlib.Path, cells: Sequence[dict]) -> None:
    nb = {
        "cells": list(cells),
        "metadata": {
            "kernelspec": {"display_name": "otfm", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1) + "\n")


# ---------------------------------------------------------------------------
# baseline_4_couplings.ipynb
# ---------------------------------------------------------------------------

BASELINE = [
    md_cell(
        "# Baseline: four couplings under identical training\n"
        "\n"
        "Reproduces the comparison of the four canonical couplings\n"
        "(``independent``, ``hungarian_exact_ot``, ``sinkhorn_sampled``,\n"
        "``sinkhorn_barycentric``) under the same MLP, optimizer, training\n"
        "budget and rollout solver."
    ),
    code_cell(
        "import jax\n"
        "import jax.numpy as jnp\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import yaml\n"
        "\n"
        "from otfm.couplings import (\n"
        "    hungarian_pairing, independent_pairing,\n"
        "    sample_pairs_from_coupling, sinkhorn_barycentric_pairing,\n"
        "    sinkhorn_coupling,\n"
        ")\n"
        "from otfm.datasets import sample_8gaussians, sample_moons\n"
        "from otfm.metrics import (\n"
        "    empirical_w2_squared_hungarian, normalized_path_energy,\n"
        "    path_energy, trajectory_curvature_metrics, velocity_variance_kernel,\n"
        ")\n"
        "from otfm.model import init_mlp_params\n"
        "from otfm.plotting import plot_generated_vs_target, plot_loss_curves, plot_low_nfe_curves\n"
        "from otfm.runtime import make_t_schedule, rollout_euler, rollout_euler_final, train_on_fixed_pairs\n"
    ),
    code_cell(
        "cfg = yaml.safe_load(open('../configs/base.yaml'))\n"
        "n = cfg['data']['n']\n"
        "eps = cfg['sinkhorn']['default_eps']\n"
        "train_steps = cfg['training']['train_steps']\n"
        "rollout_steps = cfg['rollout']['default_steps']\n"
        "widths = cfg['model']['widths']\n"
        "nfe_list = cfg['rollout']['nfe_list']\n"
    ),
    code_cell(
        "key = jax.random.PRNGKey(0)\n"
        "key, k0, k1, kpair, kinit = jax.random.split(key, 5)\n"
        "\n"
        "x0 = sample_moons(k0, n)\n"
        "x1 = sample_8gaussians(k1, n)\n"
        "\n"
        "P, _ = sinkhorn_coupling(x0, x1, epsilon=eps)\n"
        "couplings = {\n"
        "    'independent':           independent_pairing(x0, x1),\n"
        "    'hungarian_exact_ot':    hungarian_pairing(x0, x1),\n"
        "    f'sinkhorn_eps_{eps}':   sample_pairs_from_coupling(kpair, x0, x1, P),\n"
        "    'sinkhorn_barycentric':  sinkhorn_barycentric_pairing(x0, x1, P),\n"
        "}\n"
        "\n"
        "base_params = init_mlp_params(kinit, widths)\n"
        "t_schedule = make_t_schedule(seed=cfg['training']['t_schedule_seed'], steps=train_steps, n=n)\n"
    ),
    code_cell(
        "results = {}\n"
        "for name, (xa, xb) in couplings.items():\n"
        "    params_tr, losses = train_on_fixed_pairs(base_params, xa, xb, t_schedule)\n"
        "    traj = rollout_euler(params_tr, x0[:200], steps=rollout_steps)\n"
        "    results[name] = {'params': params_tr, 'losses': losses, 'traj': traj}\n"
    ),
    code_cell(
        "plot_loss_curves({name: r['losses'] for name, r in results.items()})\n"
        "plt.show()\n"
    ),
    code_cell("plot_generated_vs_target(results, x1)\nplt.show()\n"),
    code_cell(
        "import pandas as pd\n"
        "x0_eval, x1_eval = x0, x1\n"
        "w2_data = empirical_w2_squared_hungarian(x0_eval, x1_eval)\n"
        "\n"
        "rows, low_nfe_rows = [], []\n"
        "key_v = jax.random.PRNGKey(7)\n"
        "for name, r in results.items():\n"
        "    pe = path_energy(r['params'], x0_eval, steps=cfg['rollout']['pe_steps'])\n"
        "    npe = normalized_path_energy(pe, w2_data)\n"
        "    for nfe in nfe_list:\n"
        "        x_gen = rollout_euler_final(r['params'], x0_eval, steps=nfe)\n"
        "        low_nfe_rows.append({\n"
        "            'coupling': name, 'nfe': nfe,\n"
        "            'endpoint_w2_sq': empirical_w2_squared_hungarian(x_gen, x1_eval),\n"
        "        })\n"
        "    key_v, kv = jax.random.split(key_v)\n"
        "    xa, xb = couplings[name]\n"
        "    target_var = velocity_variance_kernel(\n"
        "        xa, xb, **{k: cfg['metrics']['velocity_kernel'][k]\n"
        "                   for k in ('n_time_samples', 'sigma', 'n_query', 'n_ref', 'include_time', 'time_scale')},\n"
        "        key=kv,\n"
        "    )\n"
        "    curv = trajectory_curvature_metrics(r['params'], x0_eval, steps=cfg['metrics']['curvature_steps'])\n"
        "    rows.append({\n"
        "        'coupling': name, 'PE': pe, 'NPE': npe,\n"
        "        'loss_var': float(np.var(r['losses'])),\n"
        "        'target_velocity_kernel_var': target_var,\n"
        "        **curv,\n"
        "    })\n"
        "\n"
        "summary_df = pd.DataFrame(rows).sort_values('coupling').reset_index(drop=True)\n"
        "low_nfe_df = pd.DataFrame(low_nfe_rows)\n"
        "summary_df\n"
    ),
    code_cell("plot_low_nfe_curves(low_nfe_df)\nplt.show()\n"),
]


# ---------------------------------------------------------------------------
# nfe_solver_comparison.ipynb
# ---------------------------------------------------------------------------

NFE_SOLVERS = [
    md_cell(
        "# NFE vs solver comparison\n"
        "\n"
        "Compare Euler / Heun / RK4 at matched **NFE budget** rather than matched\n"
        "step count, on a fixed trained model from the baseline. Lower endpoint\n"
        "$W_2^2$ at low NFE is the diagnostic of interest for FM."
    ),
    code_cell(
        "import jax\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import yaml\n"
        "\n"
        "from otfm.couplings import hungarian_pairing\n"
        "from otfm.datasets import sample_8gaussians, sample_moons\n"
        "from otfm.metrics import empirical_w2_squared_hungarian\n"
        "from otfm.model import init_mlp_params\n"
        "from otfm.runtime import make_t_schedule, train_on_fixed_pairs\n"
        "from otfm.solvers import SOLVERS, nfe_per_step\n"
    ),
    code_cell(
        "cfg = yaml.safe_load(open('../configs/base.yaml'))\n"
        "n = cfg['data']['n']\n"
        "widths = cfg['model']['widths']\n"
        "train_steps = cfg['training']['train_steps']\n"
        "\n"
        "key = jax.random.PRNGKey(0)\n"
        "key, k0, k1, kinit = jax.random.split(key, 4)\n"
        "x0 = sample_moons(k0, n)\n"
        "x1 = sample_8gaussians(k1, n)\n"
        "base_params = init_mlp_params(kinit, widths)\n"
        "t_schedule = make_t_schedule(seed=123, steps=train_steps, n=n)\n"
        "\n"
        "xa, xb = hungarian_pairing(x0, x1)\n"
        "params_tr, _ = train_on_fixed_pairs(base_params, xa, xb, t_schedule)\n"
    ),
    code_cell(
        "# Compare at matched NFE: pick steps so that NFE = step * cost(solver) is constant.\n"
        "target_nfes = [4, 8, 16, 32, 64]\n"
        "\n"
        "rows = []\n"
        "for solver_name, solver_fn in SOLVERS.items():\n"
        "    cost = nfe_per_step(solver_name)\n"
        "    for nfe in target_nfes:\n"
        "        steps = max(1, nfe // cost)\n"
        "        actual_nfe = steps * cost\n"
        "        x_gen = solver_fn(params_tr, x0, steps=steps)\n"
        "        w2 = empirical_w2_squared_hungarian(x_gen, x1)\n"
        "        rows.append({'solver': solver_name, 'nfe': actual_nfe, 'steps': steps, 'endpoint_w2_sq': w2})\n"
        "\n"
        "df = pd.DataFrame(rows)\n"
        "df\n"
    ),
    code_cell(
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "for s in df['solver'].unique():\n"
        "    d = df[df['solver'] == s].sort_values('nfe')\n"
        "    ax.plot(d['nfe'], d['endpoint_w2_sq'], marker='o', label=s)\n"
        "ax.set_xlabel('NFE (function evaluations)')\n"
        "ax.set_ylabel(r'endpoint $W_2^2$')\n"
        "ax.set_xscale('log', base=2)\n"
        "ax.set_yscale('log')\n"
        "ax.grid(alpha=0.2)\n"
        "ax.legend()\n"
        "plt.show()\n"
    ),
]


# ---------------------------------------------------------------------------
# multiseed_baseline.ipynb
# ---------------------------------------------------------------------------

MULTISEED = [
    md_cell(
        "# Multi-seed baseline\n"
        "\n"
        "Run the four canonical couplings across multiple seeds and aggregate\n"
        "mean ± std on key metrics."
    ),
    code_cell(
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import yaml\n"
        "\n"
        "from otfm.sweeps import SweepConfig, aggregate, run_baseline_sweep\n"
    ),
    code_cell(
        "cfg = yaml.safe_load(open('../configs/base.yaml'))\n"
        "sc = SweepConfig(\n"
        "    n=cfg['multiseed']['n'],\n"
        "    train_steps=cfg['multiseed']['train_steps'],\n"
        "    widths=cfg['model']['widths'],\n"
        "    seeds=cfg['multiseed']['seeds'],\n"
        ")\n"
        "df = run_baseline_sweep(sc)\n"
        "df.head()\n"
    ),
    code_cell(
        "metrics = ['NPE', 'endpoint_W2_sq_at_32', 'target_velocity_kernel_var', 'vel_diff_sq']\n"
        "agg = aggregate(df, group_cols=['coupling'], metrics=metrics)\n"
        "agg\n"
    ),
    code_cell(
        "fig, axes = plt.subplots(2, 2, figsize=(11, 7))\n"
        "for ax, m in zip(axes.ravel(), metrics, strict=True):\n"
        "    d = agg.sort_values('coupling')\n"
        "    ax.bar(d['coupling'], d[f'{m}_mean'], yerr=d[f'{m}_std'], capsize=4)\n"
        "    ax.set_title(m)\n"
        "    ax.tick_params(axis='x', rotation=30)\n"
        "fig.tight_layout()\n"
        "plt.show()\n"
    ),
]


# ---------------------------------------------------------------------------
# final_figures.ipynb (reads results CSVs)
# ---------------------------------------------------------------------------

FINAL = [
    md_cell(
        "# Final figures (reads ``results/full_v1.csv``)\n"
        "\n"
        "This notebook is the source of the figures shipped in the report.\n"
        "It reads the CSV produced by ``scripts/run_full_benchmark.py`` and\n"
        "produces clean publication-style plots."
    ),
    code_cell(
        "import pathlib\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from otfm.plotting import color_for, style_axes\n"
    ),
    code_cell(
        "csv_path = pathlib.Path('../results/full_v1.csv')\n"
        "if not csv_path.exists():\n"
        "    raise FileNotFoundError(\n"
        "        'Run scripts/run_full_benchmark.py first to populate results/full_v1.csv'\n"
        "    )\n"
        "df = pd.read_csv(csv_path)\n"
        "df.head()\n"
    ),
    code_cell(
        "# Headline figure: bar chart of NPE per coupling (mean ± std over seeds).\n"
        "agg = df.groupby('coupling').agg(\n"
        "    NPE_mean=('NPE', 'mean'),\n"
        "    NPE_std=('NPE', 'std'),\n"
        "    W2_mean=('endpoint_W2_sq_at_32', 'mean'),\n"
        "    W2_std=('endpoint_W2_sq_at_32', 'std'),\n"
        ").reset_index()\n"
        "agg = agg.sort_values('coupling')\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
        "for ax, (col_mean, col_std, title) in zip(\n"
        "    axes,\n"
        "    [('NPE_mean', 'NPE_std', 'NPE'), ('W2_mean', 'W2_std', r'endpoint $W_2^2$')],\n"
        "    strict=True,\n"
        "):\n"
        "    colors = [color_for(c) for c in agg['coupling']]\n"
        "    ax.bar(agg['coupling'], agg[col_mean], yerr=agg[col_std], capsize=4, color=colors)\n"
        "    ax.tick_params(axis='x', rotation=30)\n"
        "    style_axes(ax, title=title)\n"
        "fig.tight_layout()\n"
        "fig.savefig('../report/fig_headline.pdf', bbox_inches='tight')\n"
        "plt.show()\n"
    ),
]


def main():
    out = pathlib.Path(__file__).parent
    write_nb(out / "baseline_4_couplings.ipynb", BASELINE)
    write_nb(out / "nfe_solver_comparison.ipynb", NFE_SOLVERS)
    write_nb(out / "multiseed_baseline.ipynb", MULTISEED)
    write_nb(out / "final_figures.ipynb", FINAL)
    print("regenerated 4 notebooks under experiments/")


if __name__ == "__main__":
    main()
