"""Shared plotting helpers: palette, axis style, and figure layouts used in experiments."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from otfm.model import vf_apply


COUPLING_PALETTE = {
    "independent": "#7f7f7f",
    "hungarian_exact_ot": "#1f77b4",
    "exact_ot": "#1f77b4",
    "sinkhorn_sampled": "#2ca02c",
    "sinkhorn_barycentric": "#d62728",
    "ot_ind_sampled": "#9467bd",
    "ot_ind_barycentric": "#8c564b",
    "perturbed_ot": "#9467bd",
}


def color_for(name: str) -> str:
    for key, c in COUPLING_PALETTE.items():
        if key in str(name):
            return c
    return "#000000"


def style_axes(ax, *, grid: bool = True, title: str | None = None):
    if grid:
        ax.grid(alpha=0.2)
    if title is not None:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_loss_curves(losses_dict, ax=None):
    """losses_dict: {name: 1d loss array}."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for name, losses in losses_dict.items():
        ax.plot(losses, label=name, color=color_for(name))
    ax.set_xlabel("step")
    ax.set_ylabel("FM loss")
    ax.legend()
    style_axes(ax, title="Training loss (same MLP/init/optimizer/budget)")
    return ax


def plot_generated_vs_target(results, target, ncols: int = 2):
    """results: {name: {'traj': [..., x_final]}}, target: (N,2)."""
    items = list(results.items())
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, r) in zip(axes, items, strict=False):
        gen = np.asarray(r["traj"][-1])
        tgt = np.asarray(target)
        ax.scatter(tgt[:, 0], tgt[:, 1], s=8, alpha=0.25, label="target")
        ax.scatter(gen[:, 0], gen[:, 1], s=8, alpha=0.85, label="generated", color=color_for(name))
        ax.set_title(name)
        ax.axis("equal")
        ax.legend(loc="upper right")

    for ax in axes[len(items):]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_low_nfe_curves(low_nfe_df, ax=None, key_col: str = "coupling"):
    """low_nfe_df columns: [key_col, nfe, endpoint_w2_sq]."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for name in low_nfe_df[key_col].unique():
        d = low_nfe_df[low_nfe_df[key_col] == name].sort_values("nfe")
        ax.plot(d["nfe"], d["endpoint_w2_sq"], marker="o", label=name, color=color_for(str(name)))
    ax.set_xlabel("NFE")
    ax.set_ylabel(r"endpoint $W_2^2$")
    ax.legend()
    style_axes(ax, title="Low-NFE quality")
    return ax


def plot_target_field_quivers(
    couplings_dict,
    x0,
    x1,
    t_values=(0.1, 0.5, 0.9),
    max_arrows: int = 600,
    scale: float = 10.0,
    xlim=None,
    ylim=None,
    seed: int = 0,
):
    """Pre-training target field quivers for each coupling in couplings_dict."""
    rng = np.random.default_rng(seed)
    x0_np = np.asarray(x0)
    x1_np = np.asarray(x1)

    for name, (xa, xb) in couplings_dict.items():
        xa = np.asarray(xa)
        xb = np.asarray(xb)
        u = xb - xa

        n = xa.shape[0]
        idx = rng.choice(n, size=min(max_arrows, n), replace=False)
        xa_s = xa[idx]
        u_s = u[idx]

        fig, axes = plt.subplots(1, len(t_values), figsize=(6 * len(t_values), 5), sharex=True, sharey=True)
        axes = np.atleast_1d(axes)

        for ax, tval in zip(axes, t_values, strict=True):
            xt = (1.0 - tval) * xa_s + tval * (xa_s + u_s)
            ax.quiver(
                xt[:, 0],
                xt[:, 1],
                u_s[:, 0],
                u_s[:, 1],
                angles="xy",
                scale_units="xy",
                scale=scale,
                alpha=0.75,
            )
            ax.scatter(x0_np[:, 0], x0_np[:, 1], s=8, alpha=0.2, label="source")
            ax.scatter(x1_np[:, 0], x1_np[:, 1], s=8, alpha=0.2, label="target")
            ax.set_title(f"target field, t={tval:.2f}")
            ax.axis("equal")
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)

        axes[0].legend()
        fig.suptitle(f"Target vectors ({name})", y=1.02)
        fig.tight_layout()


def plot_learned_field_quivers(
    results,
    x0,
    x1,
    t_values=(0.1, 0.5, 0.9),
    grid_lim=(-7, 7),
    grid_n: int = 30,
    scale: float = 10.0,
):
    """Learned field quivers from trained params for each run in results."""
    gx = np.linspace(grid_lim[0], grid_lim[1], grid_n)
    gy = np.linspace(grid_lim[0], grid_lim[1], grid_n)
    xx, yy = np.meshgrid(gx, gy)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    x0_np = np.asarray(x0)
    x1_np = np.asarray(x1)

    for name, rec in results.items():
        params = rec["params"]
        fig, axes = plt.subplots(1, len(t_values), figsize=(6 * len(t_values), 5), sharex=True, sharey=True)
        axes = np.atleast_1d(axes)

        for ax, tval in zip(axes, t_values, strict=True):
            t = np.full((points.shape[0], 1), tval)
            v = vf_apply(params, points, t)
            u = np.asarray(v[:, 0]).reshape(xx.shape)
            w = np.asarray(v[:, 1]).reshape(yy.shape)

            ax.quiver(xx, yy, u, w, angles="xy", scale_units="xy", scale=scale, alpha=0.8)
            ax.scatter(x0_np[:, 0], x0_np[:, 1], s=8, alpha=0.2, label="source")
            ax.scatter(x1_np[:, 0], x1_np[:, 1], s=8, alpha=0.2, label="target")
            ax.set_title(f"learned field, t={tval:.2f}")
            ax.axis("equal")
            ax.set_xlim(*grid_lim)
            ax.set_ylim(*grid_lim)

        axes[0].legend()
        fig.suptitle(f"Learned vector field ({name})", y=1.02)
        fig.tight_layout()


def plot_intermediate_distributions(
    traj,
    target,
    times_show=(0.0, 0.25, 0.5, 0.75, 1.0),
    xlim=(-7, 7),
    ylim=(-7, 7),
    title: str | None = None,
):
    """Plot x_t snapshots from a trajectory list (length steps+1)."""
    steps = len(traj) - 1
    idx_show = [int(round(t * steps)) for t in times_show]

    fig, axes = plt.subplots(1, len(times_show), figsize=(4 * len(times_show), 4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    target_np = np.asarray(target)

    for ax, tval, k in zip(axes, times_show, idx_show, strict=True):
        xt = np.asarray(traj[k])
        ax.scatter(xt[:, 0], xt[:, 1], s=8, alpha=0.85, label=f"x_t")
        ax.scatter(target_np[:, 0], target_np[:, 1], s=8, alpha=0.18, label="target ref")
        ax.set_title(f"t={tval:.2f}")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")

    axes[0].legend(loc="upper right")
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_metric_curves_by_param(
    df,
    param_col: str,
    group_col: str,
    metrics,
    sort_by_param: bool = True,
    ncols: int = 3,
    figsize=(15, 8),
):
    """Generic multi-metric curve plot (e.g. epsilon/alpha sweeps)."""
    metrics = list(metrics)
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for ax, metric in zip(axes, metrics, strict=False):
        for name in df[group_col].unique():
            d = df[df[group_col] == name]
            if sort_by_param:
                d = d.sort_values(param_col)
            ax.plot(d[param_col], d[metric], marker="o", label=name, color=color_for(str(name)))
        ax.set_xlabel(param_col)
        ax.set_title(metric)
        style_axes(ax)
        ax.legend()

    for ax in axes[len(metrics):]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_pairwise_mechanism(
    df,
    pairs,
    group_col: str,
    annotate_col: str | None = None,
    ncols: int = 3,
    figsize=(16, 9),
):
    """Plot pairwise relationships for mechanism analysis."""
    pairs = list(pairs)
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for ax, item in zip(axes, pairs, strict=False):
        if len(item) == 3:
            xcol, ycol, title = item
        else:
            xcol, ycol = item
            title = f"{xcol} vs {ycol}"

        for name in df[group_col].unique():
            d = df[df[group_col] == name]
            ax.plot(d[xcol], d[ycol], marker="o", label=name, color=color_for(str(name)))
            if annotate_col is not None:
                for _, r in d.iterrows():
                    ax.annotate(f"{r[annotate_col]}", (r[xcol], r[ycol]), fontsize=8, alpha=0.8)

        ax.set_title(title)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        style_axes(ax)
        ax.legend()

    for ax in axes[len(pairs):]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_conflict_heatmap(
    heatmap,
    x_values,
    y_values,
    xlabel="x",
    ylabel="y",
    title="Heatmap",
    cbar_label="value",
    figsize=(7, 5),
):
    """Convenience heatmap for conflict/diagnostic grids."""
    heatmap = np.asarray(heatmap)
    fig = plt.figure(figsize=figsize)
    im = plt.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        extent=[min(x_values), max(x_values), min(y_values), max(y_values)],
    )
    plt.colorbar(im, label=cbar_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_jacobian_material_curves(case_curves, figsize=(12, 4)):
    """case_curves: {label: DataFrame with t,jacobian_fro_mean,acceleration_mean}."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for label, d in case_curves.items():
        axes[0].plot(d["t"], d["jacobian_fro_mean"], marker="o", label=label)
        axes[1].plot(d["t"], d["acceleration_mean"], marker="o", label=label)

    axes[0].set_title("Jacobian Frobenius mean vs t")
    axes[0].set_xlabel("t")
    style_axes(axes[0])

    axes[1].set_title("Material acceleration mean vs t")
    axes[1].set_xlabel("t")
    style_axes(axes[1])

    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig
