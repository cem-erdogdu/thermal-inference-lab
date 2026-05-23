"""Lattice-gas model with chemical potential mu.

Hamiltonian:

    H(n) = - eps * sum_{<i,j>} n_i n_j  -  mu * sum_i n_i

with n_i in {0, 1}. This is dual to the Ising model under the map
s = 2 n - 1 but exposing it directly is convenient for chemistry/
adsorption-style examples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thermal_inference_lab.core.exceptions import ValidationError
from thermal_inference_lab.core.rng import RNG
from thermal_inference_lab.core.validation import require_binary_array
from thermal_inference_lab.energy.base import EnergyModel


@dataclass(frozen=True)
class LatticeGasConfig:
    shape: tuple[int, int] = (16, 16)
    eps: float = 1.0
    mu: float = 0.0
    periodic: bool = True


class LatticeGasModel(EnergyModel):
    def __init__(self, config: LatticeGasConfig | None = None, **overrides) -> None:
        if config is None:
            config = LatticeGasConfig(**overrides)
        self.config = config

    @property
    def state_shape(self) -> tuple[int, int]:
        return self.config.shape

    @property
    def n_sites(self) -> int:
        return int(np.prod(self.config.shape))

    def random_state(self, rng: RNG) -> np.ndarray:
        return rng.integers(0, 2, size=self.state_shape).astype(np.int8)

    def _neighbour_sum(self, state: np.ndarray) -> np.ndarray:
        if self.config.periodic:
            return (
                np.roll(state, 1, axis=0)
                + np.roll(state, -1, axis=0)
                + np.roll(state, 1, axis=1)
                + np.roll(state, -1, axis=1)
            )
        padded = np.pad(state, 1, mode="constant", constant_values=0)
        return (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        )

    def total_energy(self, state: np.ndarray) -> float:
        require_binary_array(state, name="state")
        if state.shape != self.state_shape:
            raise ValidationError("shape mismatch")
        n = state.astype(np.float64)
        if self.config.periodic:
            pairs = n * (np.roll(n, -1, axis=0) + np.roll(n, -1, axis=1))
            interaction = pairs.sum()
        else:
            interaction = (n[:-1, :] * n[1:, :]).sum() + (n[:, :-1] * n[:, 1:]).sum()
        return float(-self.config.eps * interaction - self.config.mu * n.sum())

    def local_energy_delta(self, state: np.ndarray, site: tuple[int, int]) -> float:
        r, c = site
        rows, cols = self.state_shape
        current = int(state[r, c])
        flipped = 1 - current
        if self.config.periodic:
            nn = (
                int(state[(r - 1) % rows, c])
                + int(state[(r + 1) % rows, c])
                + int(state[r, (c - 1) % cols])
                + int(state[r, (c + 1) % cols])
            )
        else:
            nn = 0
            if r > 0: nn += int(state[r - 1, c])
            if r < rows - 1: nn += int(state[r + 1, c])
            if c > 0: nn += int(state[r, c - 1])
            if c < cols - 1: nn += int(state[r, c + 1])
        dn = flipped - current
        return float(-self.config.eps * dn * nn - self.config.mu * dn)

    def copy_with_flip(self, state: np.ndarray, site: tuple[int, ...]) -> np.ndarray:
        new = state.copy()
        new[site] = 1 - new[site]
        return new

    def occupancy(self, state: np.ndarray) -> float:
        return float(state.sum() / state.size)


__all__ = ["LatticeGasModel", "LatticeGasConfig"]
