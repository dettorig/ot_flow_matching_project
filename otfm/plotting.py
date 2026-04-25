"""Shared plotting helpers: palette, axis style, common figure layouts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

COUPLING_PALETTE = {
    "independent": "#7f7f7f",
    "hungarian_exact_ot": "#1f77b4",
    "sinkhorn_sampled": "#2ca02c",
    "sinkhorn_barycentric": "#d62728",
    "ot_ind_sampled": "#9467bd",
    "ot_ind_barycentric": "#8c564b",
}


def color_for(name: str) -> str:
    for key, c in COUPLING_PALETTE.items():
        if key in name:
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
    """losses_dict: {coupling_name: 1d array of losses}."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for name, losses in losses_dict.items():
        ax.plot(losses, label=name, color=color_for(name))
    ax.set_xlabel("step")
    ax.set_ylabel("FM loss")
    ax.legend()
    style_axes(ax, title="Training loss (same MLP/init/optimizer/budget)")
    return ax


def plot_generated_vs_target(results, target, n_show: int = 200, ncols: int = 2):
    """results: {name: {"traj": [..., x_final]}}, target: (N, 2)."""
    items = list(results.items())
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, r) in zip(axes, items, strict=False):
        gen = np.asarray(r["traj"][-1])
        ax.scatter(np.asarray(target)[:, 0], np.asarray(target)[:, 1], s=8, alpha=0.25, label="target")
        ax.scatter(gen[:, 0], gen[:, 1], s=8, alpha=0.85, label="generated", color=color_for(name))
        ax.set_title(name)
        ax.axis("equal")
        ax.legend(loc="upper right")
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_low_nfe_curves(low_nfe_df, ax=None, key_col: str = "coupling"):
    """low_nfe_df: columns [key_col, nfe, endpoint_w2_sq]."""
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
