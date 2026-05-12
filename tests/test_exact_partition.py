import numpy as np
import pytest

from thermal_inference_lab.core.config import IsingConfig
from thermal_inference_lab.core.exceptions import EnumerationError
from thermal_inference_lab.distributions.exact_enumeration import enumerate_ising_states
from thermal_inference_lab.distributions.partition import log_partition_function
from thermal_inference_lab.energy.ising import IsingModel


def test_enumeration_count_matches_state_space():
    model = IsingModel(IsingConfig(shape=(2, 2), J=1.0, h=0.0))
    states, energies, mags = enumerate_ising_states(model)
    assert states.shape[0] == 2 ** 4
    assert energies.shape == (16,)
    assert mags.shape == (16,)


def test_enumeration_refuses_large_lattice():
    # 5x5 = 25 spins exceeds MAX_ENUMERATION_SPINS = 24
    model = IsingModel(IsingConfig(shape=(5, 5), J=1.0))
    with pytest.raises(EnumerationError):
        enumerate_ising_states(model)


def test_log_partition_high_temperature_limit():
    # At T -> infinity, Z -> 2^N because every state is equally likely.
    model = IsingModel(IsingConfig(shape=(2, 3), J=1.0, h=0.0))
    res = log_partition_function(model, temperature=1e6)
    assert res.log_Z == pytest.approx(np.log(2 ** model.n_sites), abs=1e-3)


def test_log_partition_low_temperature_ground_state():
    # At T -> 0, Z is dominated by ground states: two aligned configs.
    model = IsingModel(IsingConfig(shape=(2, 2), J=1.0, h=0.0))
    res = log_partition_function(model, temperature=1e-3)
    # E_min = -2 * 4 (torus 2x2 has 8 bonds) = -8
    # log Z ~ log(2) - beta * E_min  (two ground states)
    log_z_expected = np.log(2.0) + 8.0 / 1e-3
    assert res.log_Z == pytest.approx(log_z_expected, rel=1e-6)
