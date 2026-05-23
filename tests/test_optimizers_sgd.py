import numpy as np
import pytest

from thermal_inference_lab.optimizers.momentum import Momentum
from thermal_inference_lab.optimizers.nesterov import Nesterov
from thermal_inference_lab.optimizers.sgd import SGD


def test_sgd_quadratic_converges():
    # min_x 0.5 ||x||^2 -> grad = x. SGD with lr=0.5 should halve x each step.
    params = {"x": np.array([1.0, -2.0, 3.0])}
    opt = SGD(lr=0.5)
    for _ in range(40):
        grads = {"x": params["x"].copy()}
        opt.step(params, grads)
    np.testing.assert_allclose(params["x"], np.zeros(3), atol=1e-6)


def test_momentum_converges_on_quadratic():
    """With a well-tuned learning rate, classical momentum drives x to zero."""
    params = {"x": np.array([5.0])}
    mom = Momentum(lr=0.01, momentum=0.9)
    for _ in range(500):
        mom.step(params, {"x": params["x"].copy()})
    assert abs(params["x"][0]) < 1e-3


def test_nesterov_converges():
    params = {"x": np.array([3.0, 4.0])}
    opt = Nesterov(lr=0.05, momentum=0.9)
    for _ in range(200):
        opt.step(params, {"x": params["x"].copy()})
    np.testing.assert_allclose(params["x"], np.zeros(2), atol=1e-3)


def test_param_grad_key_mismatch_errors():
    opt = SGD(lr=0.1)
    with pytest.raises(KeyError):
        opt.step({"a": np.zeros(1)}, {"b": np.zeros(1)})


def test_lr_must_be_positive():
    with pytest.raises(Exception):
        SGD(lr=0.0)
