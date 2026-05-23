"""Partition-function computation.

Two routes are exposed:

1. :func:`log_partition_function` (exact): uses brute-force enumeration of
   every state. Only available for tiny lattices, but ground-truth for tests.
2. :func:`log_partition_function_thermo`: a thermodynamic-integration
   *estimator* that uses sampled energy traces across a temperature ladder.
   Useful when exact enumeration is infeasible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thermal_inference_lab.core.math_utils import logsumexp
from thermal_inference_lab.core.validation import require_positive
from thermal_inference_lab.distributions.exact_enumeration import enumerate_ising_states
from thermal_inference_lab.energy.ising import IsingModel


@dataclass(frozen=True)
class LogPartitionResult:
    log_Z: float
    free_energy: float
    method: str
    n_states: int | None = None


def log_partition_function(model: IsingModel, temperature: float) -> LogPartitionResult:
    """Exact log-Z by enumeration. Refuses to run on large lattices."""
    require_positive(temperature, name="temperature")
    _, energies, _ = enumerate_ising_states(model)
    beta = 1.0 / temperature
    log_z = float(logsumexp(-beta * energies))
    free_energy = -temperature * log_z
    return LogPartitionResult(
        log_Z=log_z, free_energy=free_energy, method="exact", n_states=energies.size
    )


def log_partition_function_thermo(
    mean_energies: np.ndarray,
    temperatures: np.ndarray,
    log_z_at_infinity: float,
) -> float:
    """Estimate log Z(T_0) by thermodynamic integration from T = infinity.

    Uses the identity

        d (log Z) / d beta = - <E>_beta

    with a trapezoidal rule on the supplied (beta, <E>) ladder. The
    high-temperature anchor ``log_z_at_infinity`` is supplied by the
    caller (for an N-spin Ising model it is N * log 2).
    """
    if mean_energies.shape != temperatures.shape:
        raise ValueError("mean_energies and temperatures must align")
    order = np.argsort(temperatures)[::-1]  # decreasing T = increasing beta
    energies = mean_energies[order]
    betas = 1.0 / temperatures[order]
    log_z = log_z_at_infinity
    for i in range(1, betas.size):
        dbeta = betas[i] - betas[i - 1]
        avg = 0.5 * (energies[i] + energies[i - 1])
        log_z -= dbeta * avg
    return float(log_z)


__all__ = [
    "LogPartitionResult",
    "log_partition_function",
    "log_partition_function_thermo",
]
