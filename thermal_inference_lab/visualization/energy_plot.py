"""Energy-trace plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_energy_trace(energies: np.ndarray, ax=None, label: str | None = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(energies, lw=1.0, label=label)
    ax.set_xlabel("sweep")
    ax.set_ylabel("energy")
    if label:
        ax.legend()
    return ax


def plot_energy_histogram(energies: np.ndarray, ax=None, bins: int = 50):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(energies, bins=bins, density=True, alpha=0.7, color="steelblue")
    ax.set_xlabel("energy")
    ax.set_ylabel("density")
    return ax


__all__ = ["plot_energy_trace", "plot_energy_histogram"]
