import numpy as np
import pytest

from thermal_inference_lab.annealing.schedules import (
    ExponentialSchedule,
    LinearSchedule,
    LogarithmicSchedule,
    schedule_from_name,
)
from thermal_inference_lab.core.exceptions import ConfigurationError


def test_linear_endpoints():
    s = LinearSchedule(t_start=10.0, t_end=1.0, _n_steps=11)
    assert s(0) == pytest.approx(10.0)
    assert s(10) == pytest.approx(1.0)
    # Strict monotone decrease.
    assert all(s(i + 1) < s(i) for i in range(10))


def test_exponential_endpoints():
    s = ExponentialSchedule(t_start=10.0, t_end=0.1, _n_steps=11)
    assert s(0) == pytest.approx(10.0)
    assert s(10) == pytest.approx(0.1)


def test_logarithmic_decreases():
    s = LogarithmicSchedule(t_start=5.0, t_end=0.5, _n_steps=20)
    # Logarithmic is decreasing only after the first step.
    assert s(2) < s(0)


def test_schedule_from_name_dispatch():
    s = schedule_from_name("linear", 1.0, 0.5, 10)
    assert isinstance(s, LinearSchedule)
    s = schedule_from_name("exponential", 1.0, 0.5, 10)
    assert isinstance(s, ExponentialSchedule)
    s = schedule_from_name("logarithmic", 1.0, 0.5, 10)
    assert isinstance(s, LogarithmicSchedule)


def test_schedule_unknown_name_raises():
    with pytest.raises(ConfigurationError):
        schedule_from_name("triangular", 1.0, 0.5, 10)


def test_schedule_rejects_invalid_temperature():
    with pytest.raises(Exception):
        LinearSchedule(t_start=-1.0, t_end=0.5, _n_steps=10)
