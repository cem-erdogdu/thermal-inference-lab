import numpy as np
import pytest

from thermal_inference_lab.datasets.synthetic_spins import generate_ising_samples
from thermal_inference_lab.inference.temperature_estimation import estimate_temperature


@pytest.mark.parametrize("true_T", [2.0, 3.0])
def test_recover_temperature(true_T):
    samples = generate_ising_samples(
        shape=(6, 6),
        J=1.0,
        h=0.0,
        temperature=true_T,
        n_samples=300,
        burn_in=400,
        interval=4,
        seed=0,
    )
    fit = estimate_temperature(samples, J=1.0, h=0.0, beta_min=0.1, beta_max=2.0, n_grid=81)
    # Allow generous slack since we are using a tiny lattice.
    assert abs(fit.temperature - true_T) / true_T < 0.20
