import numpy as np
import pytest

from thermal_inference_lab.core.exceptions import ValidationError
from thermal_inference_lab.distributions.boltzmann import (
    beta_from_temperature,
    boltzmann_acceptance,
    boltzmann_log_weight,
    boltzmann_probabilities,
)


def test_beta_inversion():
    assert beta_from_temperature(2.0) == pytest.approx(0.5)


def test_log_weight_matches_minus_beta_e():
    e = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(boltzmann_log_weight(e, 0.5), -e / 0.5, atol=1e-12)


def test_acceptance_downhill_is_one():
    assert boltzmann_acceptance(-2.0, 1.0) == 1.0


def test_acceptance_uphill_in_unit_interval():
    a = boltzmann_acceptance(1.0, 1.0)
    assert 0.0 < a < 1.0
    assert a == pytest.approx(np.exp(-1.0), rel=1e-9)


def test_acceptance_rejects_invalid_temperature():
    with pytest.raises(ValidationError):
        boltzmann_acceptance(1.0, 0.0)


def test_probabilities_sum_to_one():
    energies = np.array([0.0, 1.0, 2.0, 3.0])
    p = boltzmann_probabilities(energies, 1.0)
    assert p.sum() == pytest.approx(1.0)
    # Higher energy => lower probability.
    assert np.all(np.diff(p) < 0.0)
