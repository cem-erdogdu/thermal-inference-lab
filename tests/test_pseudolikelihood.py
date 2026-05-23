import numpy as np
import pytest

from thermal_inference_lab.inference.pseudolikelihood import (
    ising_pseudolikelihood,
    ising_pseudolikelihood_gradient,
)


def _make_uniform_sample():
    rng = np.random.default_rng(0)
    return rng.choice([-1, 1], size=(20, 5, 5)).astype(np.int8)


def test_pseudolikelihood_is_finite_and_bounded():
    samples = _make_uniform_sample()
    val = ising_pseudolikelihood(samples, J=1.0, h=0.0, beta=1.0)
    assert np.isfinite(val)
    # log p in (-inf, 0]
    assert val <= 0.0


def test_pseudolikelihood_gradient_finite_difference():
    samples = _make_uniform_sample()
    h0 = 0.3
    J0 = 0.8
    beta = 1.0
    eps = 1e-5
    g = ising_pseudolikelihood_gradient(samples, J=J0, h=h0, beta=beta)
    # Numerical gradient of NEGATIVE PL.
    pl_plus_J = -ising_pseudolikelihood(samples, J=J0 + eps, h=h0, beta=beta)
    pl_minus_J = -ising_pseudolikelihood(samples, J=J0 - eps, h=h0, beta=beta)
    num_J = (pl_plus_J - pl_minus_J) / (2 * eps)
    pl_plus_h = -ising_pseudolikelihood(samples, J=J0, h=h0 + eps, beta=beta)
    pl_minus_h = -ising_pseudolikelihood(samples, J=J0, h=h0 - eps, beta=beta)
    num_h = (pl_plus_h - pl_minus_h) / (2 * eps)
    assert g["J"] == pytest.approx(num_J, abs=1e-5)
    assert g["h"] == pytest.approx(num_h, abs=1e-5)
