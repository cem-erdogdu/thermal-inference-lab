"""Synthetic data generators."""

from thermal_inference_lab.datasets.synthetic_spins import (
    generate_ising_samples,
    generate_correlated_spins,
)
from thermal_inference_lab.datasets.rbm_patterns import (
    bars_and_stripes,
    block_patterns,
)

__all__ = [
    "generate_ising_samples",
    "generate_correlated_spins",
    "bars_and_stripes",
    "block_patterns",
]
