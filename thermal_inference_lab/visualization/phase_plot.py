"""Phase-diagram plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from thermal_inference_lab.core.constants import ISING_2D_CRITICAL_TEMPERATURE


def plot_magnetization_curve(
    temperatures: np.ndarray,
    magnetizations: np.ndarray,
    ax=None,
    show_critical: bool = True,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(temperatures, magnetizations, "o-", lw=1.0)
    ax.set_xlabel("temperature")
    ax.set_ylabel("|m|")
    if show_critical:
        ax.axvline(ISING_2D_CRITICAL_TEMPERATURE, color="red", ls="--", alpha=0.5,
                   label=f"T_c (Onsager) ≈ {ISING_2D_CRITICAL_TEMPERATURE:.3f}")
        ax.legend()
    return ax


def plot_specific_heat_curve(temperatures: np.ndarray, c_values: np.ndarray, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(temperatures, c_values, "s-", lw=1.0, color="darkorange")
    ax.set_xlabel("temperature")
    ax.set_ylabel("specific heat / N")
    return ax


__all__ = ["plot_magnetization_curve", "plot_specific_heat_curve"]
