"""Estimate the temperature at which a sample was drawn.

Given a batch of equilibrium configurations and a *known* Ising model
(J, h fixed), we recover the temperature by maximising the
pseudolikelihood over a 1D scan in beta. This avoids the gradient
calculation and is robust even with a few hundred samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thermal_inference_lab.core.exceptions import InferenceError
from thermal_inference_lab.core.validation import require_positive
from thermal_inference_lab.inference.pseudolikelihood import ising_pseudolikelihood


@dataclass
class TemperatureFit:
    temperature: float
    log_pseudolikelihood: float
    betas: np.ndarray
    log_pls: np.ndarray


def estimate_temperature(
    samples: np.ndarray,
    J: float,
    h: float = 0.0,
    beta_min: float = 0.05,
    beta_max: float = 5.0,
    n_grid: int = 121,
    periodic: bool = True,
) -> TemperatureFit:
    """Scan beta on a log-uniform grid and return the maximum-PL temperature."""
    require_positive(beta_min, name="beta_min")
    require_positive(beta_max, name="beta_max")
    if beta_min >= beta_max:
        raise InferenceError("beta_min must be < beta_max")
    betas = np.exp(np.linspace(np.log(beta_min), np.log(beta_max), n_grid))
    log_pls = np.array(
        [ising_pseudolikelihood(samples, J=J, h=h, beta=b, periodic=periodic) for b in betas]
    )
    best = int(np.argmax(log_pls))
    return TemperatureFit(
        temperature=float(1.0 / betas[best]),
        log_pseudolikelihood=float(log_pls[best]),
        betas=betas,
        log_pls=log_pls,
    )


__all__ = ["estimate_temperature", "TemperatureFit"]
