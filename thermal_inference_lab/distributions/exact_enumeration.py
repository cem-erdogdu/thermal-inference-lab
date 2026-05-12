"""Brute-force enumeration of every spin configuration for tiny lattices.

Used for partition-function ground truth and for unit-testing samplers
on systems where the exact answer is known. Refuses to run when the
state space exceeds :data:`MAX_ENUMERATION_SPINS` to prevent accidental
memory blow-ups.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from thermal_inference_lab.core.constants import MAX_ENUMERATION_SPINS
from thermal_inference_lab.core.exceptions import EnumerationError
from thermal_inference_lab.energy.ising import IsingModel


def _iter_spin_configurations(shape: tuple[int, int]) -> Iterator[np.ndarray]:
    """Yield every +/-1 configuration on a lattice of ``shape`` in row-major order."""
    n = int(np.prod(shape))
    for k in range(1 << n):
        bits = np.frombuffer(k.to_bytes((n + 7) // 8, byteorder="big"), dtype=np.uint8)
        bits = np.unpackbits(bits)[-n:]
        spins = bits.astype(np.int8) * 2 - 1
        yield spins.reshape(shape)


def enumerate_ising_states(model: IsingModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (states, energies, magnetizations) for *every* configuration.

    Parameters
    ----------
    model : IsingModel
        The model to enumerate. Its ``n_sites`` must be <= MAX_ENUMERATION_SPINS.

    Returns
    -------
    states : (2^N, R, C) int8 array
    energies : (2^N,) float64 array
    magnetizations : (2^N,) float64 array (mean spin)
    """
    n = model.n_sites
    if n > MAX_ENUMERATION_SPINS:
        raise EnumerationError(
            f"Refusing to enumerate {2 ** n:,} states for {n} spins. "
            f"Cap is {MAX_ENUMERATION_SPINS} spins ({2 ** MAX_ENUMERATION_SPINS:,} states). "
            "Use Monte Carlo sampling instead."
        )
    shape = model.state_shape
    total = 1 << n
    states = np.empty((total, *shape), dtype=np.int8)
    energies = np.empty(total, dtype=np.float64)
    magnetizations = np.empty(total, dtype=np.float64)
    for i, state in enumerate(_iter_spin_configurations(shape)):
        states[i] = state
        energies[i] = model.total_energy(state)
        magnetizations[i] = state.sum() / state.size
    return states, energies, magnetizations


__all__ = ["enumerate_ising_states"]
