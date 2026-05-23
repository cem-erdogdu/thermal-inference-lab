import numpy as np
import pytest

from thermal_inference_lab.optimizers.adam import Adam
from thermal_inference_lab.optimizers.adamw import AdamW


def test_adam_converges_on_quadratic():
    params = {"x": np.array([1.0, -2.0, 3.0])}
    opt = Adam(lr=0.1)
    for _ in range(400):
        opt.step(params, {"x": params["x"].copy()})
    np.testing.assert_allclose(params["x"], np.zeros(3), atol=1e-3)


def test_adam_step_count_increments():
    opt = Adam(lr=0.1)
    params = {"x": np.zeros(2)}
    for _ in range(5):
        opt.step(params, {"x": np.ones(2)})
    assert opt.step_count == 5


def test_adamw_decay_shrinks_toward_zero():
    """With zero gradient and positive params, AdamW must drift toward 0."""
    params = {"x": np.array([1.0, 1.0, 1.0])}
    opt = AdamW(lr=0.1, weight_decay=0.5)
    for _ in range(100):
        opt.step(params, {"x": np.zeros(3)})
    assert np.all(np.abs(params["x"]) < 1.0)


def test_adam_rejects_invalid_betas():
    with pytest.raises(ValueError):
        Adam(lr=0.1, beta1=1.0)
    with pytest.raises(ValueError):
        Adam(lr=0.1, beta2=1.5)
