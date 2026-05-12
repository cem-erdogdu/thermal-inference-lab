"""Strongly-typed configuration dataclasses for experiments and models."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from thermal_inference_lab.core.exceptions import ConfigurationError


@dataclass(frozen=True)
class IsingConfig:
    """Parameters defining a 2D Ising model.

    Attributes
    ----------
    shape : (rows, cols) lattice extent.
    J : coupling strength. Positive => ferromagnetic.
    h : external field on every site.
    periodic : whether to use periodic (toroidal) boundary conditions.
    """

    shape: tuple[int, int] = (16, 16)
    J: float = 1.0
    h: float = 0.0
    periodic: bool = True

    def __post_init__(self) -> None:
        if len(self.shape) != 2:
            raise ConfigurationError(f"shape must be 2D, got {self.shape}")
        if any(s < 2 for s in self.shape):
            raise ConfigurationError(f"each dimension must be >= 2, got {self.shape}")

    @property
    def n_spins(self) -> int:
        return int(self.shape[0] * self.shape[1])


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for an MCMC sampler run."""

    n_samples: int = 1000
    burn_in: int = 1000
    interval: int = 1
    seed: int = 0
    record_energy: bool = True

    def __post_init__(self) -> None:
        if self.n_samples < 1:
            raise ConfigurationError("n_samples must be >= 1")
        if self.burn_in < 0:
            raise ConfigurationError("burn_in must be >= 0")
        if self.interval < 1:
            raise ConfigurationError("interval must be >= 1")


@dataclass(frozen=True)
class AnnealConfig:
    """Configuration for a simulated-annealing run."""

    n_steps: int = 5000
    t_start: float = 5.0
    t_end: float = 0.01
    schedule: str = "exponential"  # 'linear' | 'exponential' | 'logarithmic'
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ConfigurationError("n_steps must be >= 1")
        if self.t_start <= 0 or self.t_end <= 0:
            raise ConfigurationError("temperatures must be strictly positive")
        if self.t_end > self.t_start:
            raise ConfigurationError("t_end must be <= t_start for cooling schedules")
        if self.schedule not in {"linear", "exponential", "logarithmic"}:
            raise ConfigurationError(f"unknown schedule: {self.schedule}")


def asdict_safe(obj: Any) -> dict[str, Any]:
    """``dataclasses.asdict`` that tolerates plain dicts as well."""
    if isinstance(obj, dict):
        return dict(obj)
    return asdict(obj)


__all__ = ["IsingConfig", "SamplerConfig", "AnnealConfig", "asdict_safe"]
