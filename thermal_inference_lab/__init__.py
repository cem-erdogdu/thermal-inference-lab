"""thermal_inference_lab: NumPy-only Boltzmann sampling and energy-based inference."""

__version__ = "0.1.0"

from thermal_inference_lab.core.exceptions import (
    ThermalLabError,
    ConfigurationError,
    ValidationError,
    SamplerError,
    InferenceError,
)

__all__ = [
    "__version__",
    "ThermalLabError",
    "ConfigurationError",
    "ValidationError",
    "SamplerError",
    "InferenceError",
]
