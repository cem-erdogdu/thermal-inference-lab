"""Diagnostics computed over a sampler's output.

Includes:

* Effective sample size (ESS) from the integrated autocorrelation time.
* Geweke-style mean-shift Z-score between the first and last fraction
  of the trace.
* Rolling acceptance and energy summaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thermal_inference_lab.core.math_utils import (
    autocorrelation,
    integrated_autocorrelation_time,
)
from thermal_inference_lab.samplers.base import SamplerResult


@dataclass(frozen=True)
class SamplerDiagnostics:
    n_samples: int
    mean_energy: float
    std_energy: float
    acceptance_rate: float
    tau_int: float
    effective_sample_size: float
    geweke_z: float


def geweke_z_score(
    trace: np.ndarray, first_frac: float = 0.1, last_frac: float = 0.5
) -> float:
    """Mean-shift Z between the first ``first_frac`` and last ``last_frac``."""
    n = trace.size
    if n < 10:
        return float("nan")
    n_a = max(int(first_frac * n), 1)
    n_b = max(int(last_frac * n), 1)
    a = trace[:n_a]
    b = trace[-n_b:]
    var_a = np.var(a, ddof=1) / a.size if a.size > 1 else 1.0
    var_b = np.var(b, ddof=1) / b.size if b.size > 1 else 1.0
    denom = np.sqrt(var_a + var_b) + 1e-12
    return float((a.mean() - b.mean()) / denom)


def summarize_chain(result: SamplerResult, max_lag: int | None = None) -> SamplerDiagnostics:
    """Compute standard diagnostics for a sampler run."""
    e = result.energies
    if e.size == 0:
        return SamplerDiagnostics(
            n_samples=result.n_samples,
            mean_energy=float("nan"),
            std_energy=float("nan"),
            acceptance_rate=result.acceptance_rate,
            tau_int=float("nan"),
            effective_sample_size=float("nan"),
            geweke_z=float("nan"),
        )
    acf = autocorrelation(e, max_lag=max_lag)
    tau = integrated_autocorrelation_time(acf)
    ess = e.size / tau if tau > 0 else float("nan")
    return SamplerDiagnostics(
        n_samples=result.n_samples,
        mean_energy=float(e.mean()),
        std_energy=float(e.std(ddof=1)) if e.size > 1 else 0.0,
        acceptance_rate=result.acceptance_rate,
        tau_int=float(tau),
        effective_sample_size=float(ess),
        geweke_z=geweke_z_score(e),
    )


__all__ = ["SamplerDiagnostics", "summarize_chain", "geweke_z_score"]
