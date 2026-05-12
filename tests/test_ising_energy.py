import numpy as np
import pytest

from thermal_inference_lab.core.config import IsingConfig
from thermal_inference_lab.core.exceptions import ValidationError
from thermal_inference_lab.energy.ising import IsingModel


def test_all_aligned_state_has_minimal_energy():
    model = IsingModel(IsingConfig(shape=(4, 4), J=1.0, h=0.0))
    aligned = model.aligned_state(+1)
    # 4x4 torus has 32 bonds; H = -J * 32 = -32.
    assert model.total_energy(aligned) == pytest.approx(-32.0)


def test_external_field_shifts_energy_linearly():
    model0 = IsingModel(IsingConfig(shape=(4, 4), J=1.0, h=0.0))
    modelh = IsingModel(IsingConfig(shape=(4, 4), J=1.0, h=0.5))
    aligned = np.ones((4, 4), dtype=np.int8)
    e0 = model0.total_energy(aligned)
    eh = modelh.total_energy(aligned)
    assert eh - e0 == pytest.approx(-0.5 * 16)


def test_random_state_is_pm1():
    model = IsingModel(IsingConfig(shape=(5, 6)))
    s = model.random_state(_rng())
    assert s.shape == (5, 6)
    assert set(np.unique(s).tolist()) <= {-1, 1}


def test_energy_rejects_non_spin_arrays():
    model = IsingModel(IsingConfig(shape=(3, 3)))
    bad = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.int8)
    with pytest.raises(ValidationError):
        model.total_energy(bad)


def test_open_boundary_gives_lower_aligned_energy():
    """An open lattice has fewer bonds than a torus."""
    rows, cols = 3, 3
    torus = IsingModel(IsingConfig(shape=(rows, cols), J=1.0, periodic=True))
    open_ = IsingModel(IsingConfig(shape=(rows, cols), J=1.0, periodic=False))
    aligned = np.ones((rows, cols), dtype=np.int8)
    # torus: 2 * R*C bonds; open: R*(C-1) + C*(R-1) bonds.
    assert torus.total_energy(aligned) < open_.total_energy(aligned)


def _rng():
    from thermal_inference_lab.core.rng import RNG
    return RNG(0)
