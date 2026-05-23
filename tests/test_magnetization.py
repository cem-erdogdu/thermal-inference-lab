import numpy as np
import pytest

from thermal_inference_lab.core.config import IsingConfig
from thermal_inference_lab.distributions.observables import (
    mean_absolute_magnetization,
    mean_magnetization,
)
from thermal_inference_lab.energy.ising import IsingModel


def test_magnetization_all_up_is_one():
    m = IsingModel(IsingConfig(shape=(4, 4)))
    s = m.aligned_state(+1)
    assert m.magnetization(s) == pytest.approx(1.0)


def test_magnetization_all_down_is_minus_one():
    m = IsingModel(IsingConfig(shape=(4, 4)))
    s = m.aligned_state(-1)
    assert m.magnetization(s) == pytest.approx(-1.0)


def test_batched_mean_magnetization():
    samples = np.stack([np.ones((4, 4), dtype=np.int8), -np.ones((4, 4), dtype=np.int8)])
    assert mean_magnetization(samples) == pytest.approx(0.0)
    assert mean_absolute_magnetization(samples) == pytest.approx(1.0)
