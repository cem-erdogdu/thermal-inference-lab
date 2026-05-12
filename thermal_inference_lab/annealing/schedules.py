"""Temperature schedules for simulated annealing.

Each schedule is a callable mapping ``step -> T(step)`` with three
guarantees:

1. T(0) == t_start
2. T(n_steps - 1) == t_end  (for linear / exponential)
3. T(step) > 0 for every step.

The logarithmic schedule has the form T_n = c / log(n + 2), which is
the canonical cooling rate that guarantees (slow) convergence in
classical simulated annealing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from thermal_inference_lab.core.exceptions import ConfigurationError
from thermal_inference_lab.core.validation import require_positive


class Schedule(ABC):
    """Abstract base class for temperature schedules."""

    @abstractmethod
    def __call__(self, step: int) -> float: ...

    @property
    @abstractmethod
    def n_steps(self) -> int: ...

    def temperatures(self) -> np.ndarray:
        return np.array([self(i) for i in range(self.n_steps)], dtype=float)


@dataclass(frozen=True)
class LinearSchedule(Schedule):
    t_start: float
    t_end: float
    _n_steps: int

    def __post_init__(self) -> None:
        require_positive(self.t_start, name="t_start")
        require_positive(self.t_end, name="t_end")
        if self._n_steps < 2:
            raise ConfigurationError("n_steps must be >= 2")

    @property
    def n_steps(self) -> int:
        return self._n_steps

    def __call__(self, step: int) -> float:
        frac = step / (self._n_steps - 1)
        return float(self.t_start + frac * (self.t_end - self.t_start))


@dataclass(frozen=True)
class ExponentialSchedule(Schedule):
    """T(n) = T0 * (T_end/T0)^(n / (n_steps - 1))."""

    t_start: float
    t_end: float
    _n_steps: int

    def __post_init__(self) -> None:
        require_positive(self.t_start, name="t_start")
        require_positive(self.t_end, name="t_end")
        if self._n_steps < 2:
            raise ConfigurationError("n_steps must be >= 2")

    @property
    def n_steps(self) -> int:
        return self._n_steps

    def __call__(self, step: int) -> float:
        frac = step / (self._n_steps - 1)
        return float(self.t_start * (self.t_end / self.t_start) ** frac)


@dataclass(frozen=True)
class LogarithmicSchedule(Schedule):
    """T(n) = t_start / log(n + 2). t_end is informational."""

    t_start: float
    t_end: float
    _n_steps: int

    def __post_init__(self) -> None:
        require_positive(self.t_start, name="t_start")
        require_positive(self.t_end, name="t_end")
        if self._n_steps < 2:
            raise ConfigurationError("n_steps must be >= 2")

    @property
    def n_steps(self) -> int:
        return self._n_steps

    def __call__(self, step: int) -> float:
        # Add 2 so the first step yields T(0) = t_start / log(2).
        return float(self.t_start / np.log(step + 2.0))


def schedule_from_name(
    name: str, t_start: float, t_end: float, n_steps: int
) -> Schedule:
    """Factory for the three schedules above by string name."""
    if name == "linear":
        return LinearSchedule(t_start, t_end, n_steps)
    if name == "exponential":
        return ExponentialSchedule(t_start, t_end, n_steps)
    if name == "logarithmic":
        return LogarithmicSchedule(t_start, t_end, n_steps)
    raise ConfigurationError(f"unknown schedule: {name!r}")


__all__ = [
    "Schedule",
    "LinearSchedule",
    "ExponentialSchedule",
    "LogarithmicSchedule",
    "schedule_from_name",
]
