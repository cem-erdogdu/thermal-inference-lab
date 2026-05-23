"""Lattice configuration heatmaps."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def plot_lattice(state: np.ndarray, ax=None, title: str | None = None, cmap: str = "RdBu"):
    """Render a single 2D spin configuration as a heatmap."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(state, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    return im


def plot_lattice_grid(
    states: Iterable[np.ndarray],
    titles: Iterable[str] | None = None,
    n_cols: int = 4,
    figsize: tuple[float, float] | None = None,
):
    """Render a grid of lattice configurations."""
    states = list(states)
    titles = list(titles) if titles is not None else [None] * len(states)
    n = len(states)
    n_rows = (n + n_cols - 1) // n_cols
    figsize = figsize or (3 * n_cols, 3 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    for i, (state, title) in enumerate(zip(states, titles)):
        ax = axes[i // n_cols, i % n_cols]
        plot_lattice(state, ax=ax, title=title)
    for j in range(n, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")
    fig.tight_layout()
    return fig


__all__ = ["plot_lattice", "plot_lattice_grid"]
