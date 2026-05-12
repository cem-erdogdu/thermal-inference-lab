"""Abstract energy-model interface.

Every concrete energy model in this package implements :class:`EnergyModel`.
The interface deliberately stays minimal: samplers and inference code only
ever need ``total_energy``, ``local_energy_delta`` (for efficient single-spin
proposals), and ``random_state`` (for chain initialisation).

Sub-classes may add domain-specific methods (e.g., ``magnetization`` for
spin models), but the sampler protocol is anchored here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from thermal_inference_lab.core.rng import RNG


class EnergyModel(ABC):
    """Abstract base for any system whose Boltzmann distribution we sample."""

    @property
    @abstractmethod
    def state_shape(self) -> tuple[int, ...]:
        """Shape of a single configuration array."""

    @property
    @abstractmethod
    def n_sites(self) -> int:
        """Number of degrees of freedom (lattice sites)."""

    @abstractmethod
    def total_energy(self, state: np.ndarray) -> float:
        """Compute the total Hamiltonian of a configuration."""

    @abstractmethod
    def local_energy_delta(self, state: np.ndarray, site: tuple[int, ...]) -> float:
        """ΔE for flipping/changing the value at ``site``.

        Implementations must guarantee that

            E(state_after_flip) == E(state_before_flip) + local_energy_delta(...)

        up to floating-point noise. This invariant is exercised in the
        test suite and any deviation indicates a bug.
        """

    @abstractmethod
    def random_state(self, rng: RNG) -> np.ndarray:
        """Return an uncorrelated random configuration."""

    # ----------------------------------------------------------- common
    def iter_sites(self) -> Iterable[tuple[int, ...]]:
        """Yield site indices in row-major order. Helpful for Gibbs sweeps."""
        ranges = [range(s) for s in self.state_shape]
        from itertools import product

        return product(*ranges)

    def copy_with_flip(self, state: np.ndarray, site: tuple[int, ...]) -> np.ndarray:
        """Return a copy of ``state`` with a single site flipped (default: x -> -x)."""
        new = state.copy()
        new[site] = -new[site]
        return new

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"{self.__class__.__name__}(shape={self.state_shape})"


__all__ = ["EnergyModel"]
