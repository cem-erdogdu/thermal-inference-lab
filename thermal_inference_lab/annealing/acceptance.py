"""Acceptance criteria for annealing moves.

Right now we expose only the standard Metropolis acceptance rule; other
rules (e.g., Tsallis acceptance for generalised simulated annealing)
can be added here later without touching the annealing driver.
"""

from __future__ import annotations

import numpy as np

from thermal_inference_lab.core.validation import require_positive


def metropolis_accept(delta_e: float, temperature: float, u: float) -> bool:
    """Return True iff a uniformly drawn ``u`` accepts the move ΔE at T."""
    if delta_e <= 0.0:
        return True
    require_positive(temperature, name="temperature")
    return bool(u < np.exp(max(-delta_e / temperature, -700.0)))


__all__ = ["metropolis_accept"]
