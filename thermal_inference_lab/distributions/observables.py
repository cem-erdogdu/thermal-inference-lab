"""Thermodynamic observables derived from sampled trajectories.

Given a sample (or sample trace) of states and the corresponding energy
trace from a sampler, these helpers return the standard observables
used to diagnose phase transitions in spin systems.
"""

from __future__ import annotations

import numpy as np

from thermal_inference_lab.core.validation import require_positive


def mean_energy(energy_trace: np.ndarray) -> float:
    return float(np.mean(energy_trace))


def mean_magnetization(samples: np.ndarray) -> float:
    """Mean of m = (1/N) * sum_i s_i across the sample dimension (axis 0)."""
    per_sample_mag = samples.reshape(samples.shape[0], -1).mean(axis=1)
    return float(per_sample_mag.mean())


def mean_absolute_magnetization(samples: np.ndarray) -> float:
    per_sample_mag = samples.reshape(samples.shape[0], -1).mean(axis=1)
    return float(np.abs(per_sample_mag).mean())


def specific_heat(energy_trace: np.ndarray, temperature: float, n_sites: int) -> float:
    """C/N = Var(E) / (N * T^2)."""
    require_positive(temperature, name="temperature")
    var_e = float(np.var(energy_trace))
    return var_e / (n_sites * temperature * temperature)


def susceptibility(samples: np.ndarray, temperature: float) -> float:
    """Magnetic susceptibility chi = N * Var(m) / T."""
    require_positive(temperature, name="temperature")
    n = samples.reshape(samples.shape[0], -1).shape[1]
    per_sample_mag = samples.reshape(samples.shape[0], -1).mean(axis=1)
    return float(n * np.var(per_sample_mag) / temperature)


def binder_cumulant(samples: np.ndarray) -> float:
    """U_4 = 1 - <m^4> / (3 <m^2>^2). Dimensionless, equals 2/3 at criticality."""
    per_sample_mag = samples.reshape(samples.shape[0], -1).mean(axis=1)
    m2 = float(np.mean(per_sample_mag ** 2))
    m4 = float(np.mean(per_sample_mag ** 4))
    if m2 < 1e-30:
        return 0.0
    return float(1.0 - m4 / (3.0 * m2 * m2))


def correlation_length_proxy(samples: np.ndarray) -> float:
    """Crude estimate of correlation length via the variance of magnetization.

    Returns ``Var(m)`` times the linear lattice size. This is only a
    proxy: it tracks correlation length scaling but is not the true
    second-moment correlator. Sufficient for phase-diagram sketches.
    """
    arr = samples.reshape(samples.shape[0], -1)
    return float(arr.shape[1] ** 0.5 * np.var(arr.mean(axis=1)))


__all__ = [
    "mean_energy",
    "mean_magnetization",
    "mean_absolute_magnetization",
    "specific_heat",
    "susceptibility",
    "binder_cumulant",
    "correlation_length_proxy",
]
