import numpy as np
import pytest

from thermal_inference_lab.core.math_utils import sigmoid, softplus
from thermal_inference_lab.core.rng import RNG
from thermal_inference_lab.energy.rbm import RBMEnergy


def test_free_energy_matches_definition():
    rng = RNG(0)
    rbm = RBMEnergy(n_visible=4, n_hidden=3, rng=rng, weight_scale=0.5)
    v = np.array([1, 0, 1, 0])
    expected = -(rbm.params.a @ v) - softplus(v @ rbm.params.W + rbm.params.b).sum()
    assert rbm.free_energy(v) == pytest.approx(float(expected), rel=1e-10)


def test_conditional_probabilities_match_sigmoid():
    rng = RNG(0)
    rbm = RBMEnergy(n_visible=3, n_hidden=2, rng=rng, weight_scale=0.3)
    v = np.array([[1, 0, 1]])
    p_h = rbm.prob_hidden_given_visible(v)
    expected = sigmoid(v @ rbm.params.W + rbm.params.b)
    np.testing.assert_allclose(p_h, expected, atol=1e-10)


def test_sample_shapes():
    rng = RNG(0)
    rbm = RBMEnergy(n_visible=5, n_hidden=4, rng=rng)
    v = np.array([[1, 0, 1, 1, 0]])
    h = rbm.sample_hidden_given_visible(v, rng)
    assert h.shape == (1, 4)
    assert set(np.unique(h).tolist()) <= {0, 1}


def test_free_energy_gradients_shapes():
    rng = RNG(0)
    rbm = RBMEnergy(n_visible=4, n_hidden=3, rng=rng)
    V = np.array([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=np.float64)
    grads = rbm.free_energy_gradients(V)
    assert grads["W"].shape == (4, 3)
    assert grads["a"].shape == (4,)
    assert grads["b"].shape == (3,)
