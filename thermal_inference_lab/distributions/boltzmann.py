"""Boltzmann distribution helpers.

We work with log-weights everywhere to keep the arithmetic stable at
the low temperatures encountered during annealing. Acceptance ratios
are computed via ``min(1, exp(-beta * dE))`` while clamping the
exponent argument to avoid overflow.
"""

from __future__ import annotations

import numpy as np

from thermal_inference_lab.core.constants import KB
from thermal_inference_lab.core.validation import require_positive


def beta_from_temperature(temperature: float) -> float:
    """β = 1 / (k_B T) in reduced units (k_B = 1)."""
    require_positive(temperature, name="temperature")
    return 1.0 / (KB * temperature)


def boltzmann_log_weight(energy: float | np.ndarray, temperature: float) -> np.ndarray:
    """Unnormalised log-weight log(exp(-βE)) = -βE."""
    require_positive(temperature, name="temperature")
    return -np.asarray(energy, dtype=float) / (KB * temperature)


def boltzmann_weight(energy: float | np.ndarray, temperature: float) -> np.ndarray:
    """Unnormalised Boltzmann weight exp(-βE).

    The energies are shifted by their minimum before exponentiation so
    the largest weight equals 1 and overflow cannot occur. The returned
    array is therefore proportional to (but not equal to) exp(-βE).
    """
    log_w = boltzmann_log_weight(energy, temperature)
    log_w = np.asarray(log_w)
    return np.exp(log_w - log_w.max())


def boltzmann_acceptance(delta_e: float, temperature: float) -> float:
    """Metropolis-Hastings acceptance probability for ΔE at temperature T.

    Returns ``min(1, exp(-βΔE))`` while guarding against overflow.
    """
    require_positive(temperature, name="temperature")
    if delta_e <= 0.0:
        return 1.0
    exponent = -delta_e / (KB * temperature)
    # exponent <= 0 here, so exp is in (0, 1]; clamp to avoid underflow.
    return float(np.exp(max(exponent, -700.0)))


def boltzmann_probabilities(
    energies: np.ndarray, temperature: float
) -> np.ndarray:
    """Properly normalised p_i = exp(-β E_i) / Z over a finite set."""
    log_w = boltzmann_log_weight(energies, temperature)
    log_w = log_w - log_w.max()
    w = np.exp(log_w)
    return w / w.sum()


__all__ = [
    "beta_from_temperature",
    "boltzmann_log_weight",
    "boltzmann_weight",
    "boltzmann_acceptance",
    "boltzmann_probabilities",
]
