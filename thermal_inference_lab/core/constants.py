"""Physical and numerical constants used across the package.

We work in *reduced units* where the Boltzmann constant k_B = 1 by
convention. This is standard in computational statistical mechanics and
removes a stray multiplicative constant from every Boltzmann factor.
"""

from __future__ import annotations

import numpy as np

# --- Reduced-unit physical constants -----------------------------------------
KB: float = 1.0
"""Boltzmann constant in reduced units."""

# Critical temperature of the 2D Ising model on an infinite square lattice
# with J=1 in reduced units (Onsager 1944): T_c = 2 / ln(1 + sqrt(2)).
ISING_2D_CRITICAL_TEMPERATURE: float = 2.0 / np.log(1.0 + np.sqrt(2.0))

# --- Numerical thresholds -----------------------------------------------------
EPS: float = 1e-12
"""Generic floor used to guard logarithms and divisions."""

LOG_EPS: float = -30.0
"""Lower clamp for log-probabilities to avoid -inf in numerics."""

MAX_ENUMERATION_SPINS: int = 24
"""Hard cap on the number of spins for which exact enumeration is allowed.

24 spins => 2**24 = 16,777,216 configurations, still tractable in memory
if stored as packed bits but slow. Anything larger is rejected to prevent
accidental memory blow-ups.
"""

DEFAULT_BURN_IN: int = 1000
DEFAULT_INTERVAL: int = 10

__all__ = [
    "KB",
    "ISING_2D_CRITICAL_TEMPERATURE",
    "EPS",
    "LOG_EPS",
    "MAX_ENUMERATION_SPINS",
    "DEFAULT_BURN_IN",
    "DEFAULT_INTERVAL",
]
