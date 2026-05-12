"""Precondition checks used throughout the library.

Validation is centralised here so that error messages are uniform and so
that the calling site stays readable. All routines raise
:class:`ValidationError` (a subclass of ``ThermalLabError``).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from thermal_inference_lab.core.exceptions import ValidationError


def require_spin_array(state: np.ndarray, *, ndim: int | None = None, name: str = "state") -> None:
    """Verify that ``state`` is a numpy array of +/-1 spins.

    Optionally enforces an exact dimensionality (e.g., 2 for a 2D Ising
    lattice). The dtype is intentionally not constrained: integer and
    floating point spin arrays are both common in the literature.
    """
    if not isinstance(state, np.ndarray):
        raise ValidationError(f"{name} must be a numpy.ndarray, got {type(state).__name__}")
    if ndim is not None and state.ndim != ndim:
        raise ValidationError(f"{name} must be {ndim}-dimensional, got ndim={state.ndim}")
    if state.size == 0:
        raise ValidationError(f"{name} must not be empty")
    unique = np.unique(state)
    if not np.all(np.isin(unique, (-1, 1))):
        bad = unique[~np.isin(unique, (-1, 1))]
        raise ValidationError(
            f"{name} must only contain -1 and +1 spins; found {bad.tolist()[:5]}"
        )


def require_binary_array(state: np.ndarray, *, name: str = "state") -> None:
    """Verify that ``state`` is a numpy array of 0/1 values."""
    if not isinstance(state, np.ndarray):
        raise ValidationError(f"{name} must be a numpy.ndarray, got {type(state).__name__}")
    unique = np.unique(state)
    if not np.all(np.isin(unique, (0, 1))):
        bad = unique[~np.isin(unique, (0, 1))]
        raise ValidationError(
            f"{name} must only contain 0 and 1 values; found {bad.tolist()[:5]}"
        )


def require_probability(p: float, *, name: str = "p") -> None:
    if not np.isfinite(p):
        raise ValidationError(f"{name} must be finite, got {p}")
    if not (0.0 <= p <= 1.0):
        raise ValidationError(f"{name} must lie in [0, 1], got {p}")


def require_positive(value: float, *, name: str) -> None:
    if not np.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value}")
    if value <= 0.0:
        raise ValidationError(f"{name} must be strictly positive, got {value}")


def require_non_negative(value: float, *, name: str) -> None:
    if not np.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value}")
    if value < 0.0:
        raise ValidationError(f"{name} must be non-negative, got {value}")


def require_finite(array: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} contains non-finite entries")


def require_shape(array: np.ndarray, expected: tuple[int, ...], *, name: str) -> None:
    if array.shape != expected:
        raise ValidationError(f"{name} must have shape {expected}, got {array.shape}")


def require_in(value, choices: Iterable, *, name: str) -> None:
    choices = list(choices)
    if value not in choices:
        raise ValidationError(f"{name} must be one of {choices}, got {value!r}")


__all__ = [
    "require_spin_array",
    "require_binary_array",
    "require_probability",
    "require_positive",
    "require_non_negative",
    "require_finite",
    "require_shape",
    "require_in",
]
