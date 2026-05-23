"""Toy crystal-defect energy model.

We model a 2D crystal as a binary occupancy lattice in {0, 1}: 0 = atom
in normal site, 1 = vacancy (defect). The energy is

    E = epsilon_form * N_def  +  J_int * sum_{<i,j>} (d_i XOR d_j)

where ``epsilon_form`` is the formation energy of a single defect and
``J_int`` penalises defect/non-defect interfaces (like a surface
tension). This is sufficient to demonstrate annealing and Boltzmann
sampling on a non-spin system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thermal_inference_lab.core.exceptions import ValidationError
from thermal_inference_lab.core.rng import RNG
from thermal_inference_lab.core.validation import require_binary_array
from thermal_inference_lab.energy.base import EnergyModel


@dataclass(frozen=True)
class CrystalDefectConfig:
    shape: tuple[int, int] = (16, 16)
    epsilon_form: float = 1.5
    J_int: float = 0.5
    periodic: bool = True


class CrystalDefectModel(EnergyModel):
    """Binary occupancy lattice with formation energy + interface cost."""

    def __init__(self, config: CrystalDefectConfig | None = None, **overrides) -> None:
        if config is None:
            config = CrystalDefectConfig(**overrides)
        self.config = config

    @property
    def state_shape(self) -> tuple[int, int]:
        return self.config.shape

    @property
    def n_sites(self) -> int:
        return int(np.prod(self.config.shape))

    def random_state(self, rng: RNG) -> np.ndarray:
        return rng.integers(0, 2, size=self.state_shape).astype(np.int8)

    def total_energy(self, state: np.ndarray) -> float:
        require_binary_array(state, name="state")
        if state.shape != self.state_shape:
            raise ValidationError(
                f"state shape {state.shape} != model shape {self.state_shape}"
            )
        s = state.astype(np.int32)
        e_form = self.config.epsilon_form * s.sum()
        if self.config.periodic:
            interfaces = (
                (s != np.roll(s, -1, axis=0)).sum() + (s != np.roll(s, -1, axis=1)).sum()
            )
        else:
            interfaces = (s[:-1, :] != s[1:, :]).sum() + (s[:, :-1] != s[:, 1:]).sum()
        return float(e_form + self.config.J_int * interfaces)

    def local_energy_delta(self, state: np.ndarray, site: tuple[int, int]) -> float:
        r, c = site
        rows, cols = self.state_shape
        s = state
        current = int(s[r, c])
        flipped = 1 - current
        if self.config.periodic:
            neighbours = [
                s[(r - 1) % rows, c],
                s[(r + 1) % rows, c],
                s[r, (c - 1) % cols],
                s[r, (c + 1) % cols],
            ]
        else:
            neighbours = []
            if r > 0:
                neighbours.append(s[r - 1, c])
            if r < rows - 1:
                neighbours.append(s[r + 1, c])
            if c > 0:
                neighbours.append(s[r, c - 1])
            if c < cols - 1:
                neighbours.append(s[r, c + 1])
        d_form = self.config.epsilon_form * (flipped - current)
        before_iface = sum(int(current != n) for n in neighbours)
        after_iface = sum(int(flipped != n) for n in neighbours)
        d_iface = self.config.J_int * (after_iface - before_iface)
        return float(d_form + d_iface)

    def copy_with_flip(self, state: np.ndarray, site: tuple[int, ...]) -> np.ndarray:
        """Flip 0<->1 instead of the default +/-1 sign flip."""
        new = state.copy()
        new[site] = 1 - new[site]
        return new

    def defect_density(self, state: np.ndarray) -> float:
        return float(state.sum() / state.size)


__all__ = ["CrystalDefectModel", "CrystalDefectConfig"]
