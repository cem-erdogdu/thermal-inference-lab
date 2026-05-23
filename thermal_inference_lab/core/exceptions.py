"""Custom exception hierarchy for thermal_inference_lab.

All library-specific failures inherit from ``ThermalLabError`` so callers
can catch them uniformly without masking unrelated bugs in user code.
"""

from __future__ import annotations


class ThermalLabError(Exception):
    """Base class for every error raised by the library."""


class ConfigurationError(ThermalLabError):
    """Raised when a user-provided configuration object is invalid."""


class ValidationError(ThermalLabError):
    """Raised when an input array or value fails a precondition check."""


class SamplerError(ThermalLabError):
    """Raised on internal sampler failures (e.g., divergent acceptance)."""


class InferenceError(ThermalLabError):
    """Raised when an inference routine cannot converge or produce a fit."""


class EnumerationError(ThermalLabError):
    """Raised when exact enumeration is attempted on a too-large state space."""
