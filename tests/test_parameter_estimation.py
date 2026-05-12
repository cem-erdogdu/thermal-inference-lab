import numpy as np
import pytest

from thermal_inference_lab.datasets.synthetic_spins import generate_ising_samples
from thermal_inference_lab.inference.parameter_estimation import estimate_ising_parameters


def test_recover_J_when_h_zero():
    samples = generate_ising_samples(
        shape=(6, 6),
        J=1.0,
        h=0.0,
        temperature=2.6,
        n_samples=300,
        burn_in=400,
        interval=4,
        seed=0,
    )
    fit = estimate_ising_parameters(
        samples, beta=1.0 / 2.6, init_J=0.0, init_h=0.0, n_iter=400, tol=1e-7
    )
    assert abs(fit.J - 1.0) < 0.15
    assert abs(fit.h - 0.0) < 0.10


def test_recover_h_with_known_J():
    samples = generate_ising_samples(
        shape=(6, 6),
        J=1.0,
        h=0.4,
        temperature=3.0,
        n_samples=300,
        burn_in=400,
        interval=4,
        seed=1,
    )
    fit = estimate_ising_parameters(
        samples, beta=1.0 / 3.0, init_J=0.5, init_h=0.0, n_iter=400, tol=1e-7
    )
    assert abs(fit.h - 0.4) < 0.15
    assert abs(fit.J - 1.0) < 0.2


def test_fit_returns_loss_history():
    samples = generate_ising_samples(
        shape=(4, 4), temperature=3.0, n_samples=80, burn_in=100, seed=0
    )
    fit = estimate_ising_parameters(samples, beta=1 / 3.0, n_iter=50)
    assert fit.loss_history.size <= 50
    assert np.all(np.isfinite(fit.loss_history))
