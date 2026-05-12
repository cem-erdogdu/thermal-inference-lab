"""Sampler-diagnostics plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_autocorrelation(acf: np.ndarray, ax=None, label: str | None = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(acf, lw=1.0, label=label)
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_xlabel("lag")
    ax.set_ylabel("autocorrelation")
    if label:
        ax.legend()
    return ax


def plot_acceptance_rolling(acceptance_per_sweep: np.ndarray, window: int = 50, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    if acceptance_per_sweep.size >= window:
        kernel = np.ones(window) / window
        rolling = np.convolve(acceptance_per_sweep, kernel, mode="valid")
        ax.plot(rolling)
    else:
        ax.plot(acceptance_per_sweep)
    ax.set_xlabel("sweep")
    ax.set_ylabel("acceptance rate")
    ax.set_ylim(0, 1)
    return ax


__all__ = ["plot_autocorrelation", "plot_acceptance_rolling"]
