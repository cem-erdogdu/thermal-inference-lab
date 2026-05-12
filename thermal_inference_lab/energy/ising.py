"""2D Ising model with periodic boundary conditions.

Hamiltonian (reduced units, J and h constants over the lattice):

    H(s) = -J * sum_{<i,j>} s_i s_j  -  h * sum_i s_i

The sum over <i,j> is over nearest-neighbour pairs counted once. For a
toroidal lattice of shape (R, C) this evaluates to a fully vectorised
expression using ``np.roll`` over the two axes.

The local energy delta for flipping site (r, c) is

    ΔE = 2 * s_{r,c} * (J * sum_nn + h)

which is the standard textbook result (see Newman & Barkema 1999).
"""

from __future__ import annotations

import numpy as np

from thermal_inference_lab.core.config import IsingConfig
from thermal_inference_lab.core.exceptions import ValidationError
from thermal_inference_lab.core.rng import RNG
from thermal_inference_lab.core.validation import require_spin_array
from thermal_inference_lab.energy.base import EnergyModel


class IsingModel(EnergyModel):
    """2D Ising model on a square lattice."""

    def __init__(self, config: IsingConfig | None = None, **overrides) -> None:
        if config is None:
            config = IsingConfig(**overrides)
        elif overrides:
            raise ValueError("pass either config or kwargs, not both")
        self.config = config

    # --------------------------------------------------------- properties
    @property
    def state_shape(self) -> tuple[int, int]:
        return self.config.shape

    @property
    def n_sites(self) -> int:
        return self.config.n_spins

    @property
    def J(self) -> float:
        return self.config.J

    @property
    def h(self) -> float:
        return self.config.h

    @property
    def periodic(self) -> bool:
        return self.config.periodic

    # ---------------------------------------------------------- factories
    def random_state(self, rng: RNG) -> np.ndarray:
        """Uniform random +/-1 spins."""
        state = rng.integers(0, 2, size=self.state_shape).astype(np.int8) * 2 - 1
        return state

    def aligned_state(self, value: int = 1) -> np.ndarray:
        if value not in (-1, 1):
            raise ValueError("aligned spins must be +/-1")
        return np.full(self.state_shape, value, dtype=np.int8)

    # ------------------------------------------------------ neighbour sum
    def _neighbour_sum(self, state: np.ndarray) -> np.ndarray:
        """Vectorised sum of the 4 nearest-neighbour spins at every site."""
        if self.periodic:
            return (
                np.roll(state, 1, axis=0)
                + np.roll(state, -1, axis=0)
                + np.roll(state, 1, axis=1)
                + np.roll(state, -1, axis=1)
            )
        # Open boundary: zero-pad with one extra row/col of zero "ghost" spins.
        padded = np.pad(state, 1, mode="constant", constant_values=0)
        center = padded[1:-1, 1:-1]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        del center  # only padding-derived neighbours used
        return up + down + left + right

    # -------------------------------------------------------------- energy
    def total_energy(self, state: np.ndarray) -> float:
        require_spin_array(state, ndim=2)
        if state.shape != self.state_shape:
            raise ValidationError(
                f"state shape {state.shape} != model shape {self.state_shape}"
            )
        s = state.astype(np.float64)
        # Each pair is counted once: take only "right" and "down" neighbours.
        if self.periodic:
            interaction = s * (np.roll(s, -1, axis=0) + np.roll(s, -1, axis=1))
        else:
            right_pair = s[:, :-1] * s[:, 1:]
            down_pair = s[:-1, :] * s[1:, :]
            interaction = right_pair.sum() + down_pair.sum()
            return float(-self.J * interaction - self.h * s.sum())
        return float(-self.J * interaction.sum() - self.h * s.sum())

    def local_energy_delta(self, state: np.ndarray, site: tuple[int, int]) -> float:
        """ΔE = E(after flip) - E(before flip) for a single-spin flip."""
        r, c = site
        s_rc = state[r, c]
        nn = self._neighbour_at(state, r, c)
        return float(2.0 * s_rc * (self.J * nn + self.h))

    def _neighbour_at(self, state: np.ndarray, r: int, c: int) -> int:
        rows, cols = self.state_shape
        if self.periodic:
            up = state[(r - 1) % rows, c]
            down = state[(r + 1) % rows, c]
            left = state[r, (c - 1) % cols]
            right = state[r, (c + 1) % cols]
        else:
            up = state[r - 1, c] if r > 0 else 0
            down = state[r + 1, c] if r < rows - 1 else 0
            left = state[r, c - 1] if c > 0 else 0
            right = state[r, c + 1] if c < cols - 1 else 0
        return int(up + down + left + right)

    # ------------------------------------------------------- observables
    def magnetization(self, state: np.ndarray) -> float:
        require_spin_array(state, ndim=2)
        return float(state.sum() / state.size)

    def absolute_magnetization(self, state: np.ndarray) -> float:
        return float(np.abs(state.sum()) / state.size)

    def energy_per_site(self, state: np.ndarray) -> float:
        return self.total_energy(state) / self.n_sites

    # --------------------------- vectorised local fields for whole lattice
    def local_fields(self, state: np.ndarray) -> np.ndarray:
        """Effective local field h_i = J * sum_nn + h for every site.

        Useful for Gibbs sampling and pseudolikelihood inference: the
        conditional distribution of spin i given the rest is
        p(s_i = +1 | rest) = sigmoid(2 * (J*nn + h) / T).
        """
        return self.J * self._neighbour_sum(state) + self.h

    def __repr__(self) -> str:  # pragma: no cover
        return f"IsingModel(shape={self.state_shape}, J={self.J}, h={self.h})"


__all__ = ["IsingModel"]
