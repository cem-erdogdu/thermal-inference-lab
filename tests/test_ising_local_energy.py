"""The local-energy-delta invariant: flipping site i changes E by exactly ΔE."""

import numpy as np
import pytest

from thermal_inference_lab.core.config import IsingConfig
from thermal_inference_lab.core.rng import RNG
from thermal_inference_lab.energy.ising import IsingModel


@pytest.mark.parametrize("shape", [(4, 4), (5, 7)])
@pytest.mark.parametrize("h", [0.0, 0.3])
@pytest.mark.parametrize("periodic", [True, False])
def test_local_energy_delta_matches_recompute(shape, h, periodic):
    model = IsingModel(IsingConfig(shape=shape, J=1.0, h=h, periodic=periodic))
    rng = RNG(123)
    state = model.random_state(rng)
    for _ in range(40):
        r = int(rng.integers(0, shape[0]))
        c = int(rng.integers(0, shape[1]))
        e_before = model.total_energy(state)
        d_e = model.local_energy_delta(state, (r, c))
        state[r, c] = -state[r, c]
        e_after = model.total_energy(state)
        assert e_after - e_before == pytest.approx(d_e, abs=1e-9)


def test_delta_aligned_h_zero():
    # All +1 spins; flipping one breaks 4 satisfied bonds: ΔE = +8.
    model = IsingModel(IsingConfig(shape=(4, 4), J=1.0, h=0.0))
    state = np.ones((4, 4), dtype=np.int8)
    assert model.local_energy_delta(state, (1, 1)) == pytest.approx(8.0)
