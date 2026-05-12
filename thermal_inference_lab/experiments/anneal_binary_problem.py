"""Solve a Max-Cut-style binary problem with simulated annealing on an Ising rep.

A weighted graph G = (V, E, w) has Max-Cut SDP-equivalent Ising energy

    H(s) = - sum_{(i,j) in E} w_ij * s_i * s_j,    s_i in {-1, +1}.

Minimising H is equivalent (up to a constant) to maximising the cut.
We construct a small random graph, encode it as a fully-connected
Ising model with non-uniform couplings, and run simulated annealing.

The point of this experiment is to demonstrate the *generic*
:class:`SimulatedAnnealing` driver on a Hamiltonian that is not the
nearest-neighbour Ising used elsewhere.
"""

from __future__ import annotations

import argparse

import numpy as np

from thermal_inference_lab.annealing.schedules import ExponentialSchedule
from thermal_inference_lab.annealing.simulated_annealing import SimulatedAnnealing
from thermal_inference_lab.core.rng import RNG
from thermal_inference_lab.energy.base import EnergyModel


class WeightedIsingGraph(EnergyModel):
    """Fully-connected Ising model with an arbitrary coupling matrix W."""

    def __init__(self, W: np.ndarray) -> None:
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("W must be a square matrix")
        # Symmetrise and zero the diagonal.
        self.W = 0.5 * (W + W.T)
        np.fill_diagonal(self.W, 0.0)
        self._n = self.W.shape[0]

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self._n,)

    @property
    def n_sites(self) -> int:
        return self._n

    def random_state(self, rng: RNG) -> np.ndarray:
        return (rng.integers(0, 2, size=self._n).astype(np.int8) * 2 - 1)

    def total_energy(self, state: np.ndarray) -> float:
        s = state.astype(np.float64)
        return float(-0.5 * s @ self.W @ s)

    def local_energy_delta(self, state: np.ndarray, site: tuple[int, ...]) -> float:
        (i,) = site
        return float(2.0 * state[i] * (self.W[i] @ state))


def main(n: int = 24, n_steps: int = 30000, seed: int = 0) -> None:
    rng = RNG(seed)
    # Random sparse weighted graph (Erdos-Renyi).
    p = 0.3
    raw = (rng.random((n, n)) < p).astype(np.float64) * rng.uniform(-1.0, 1.0, size=(n, n))
    W = 0.5 * (raw + raw.T)
    np.fill_diagonal(W, 0.0)
    model = WeightedIsingGraph(W)

    schedule = ExponentialSchedule(t_start=5.0, t_end=1e-3, _n_steps=n_steps)
    sa = SimulatedAnnealing(model=model, schedule=schedule, rng=seed)
    result = sa.run()
    print(f"n={n} nodes, edges drawn ~ {int((W != 0).sum() // 2)}")
    print(f"best energy : {result.best_energy:.6f}")
    print(f"final energy: {result.final_energy:.6f}")
    print(f"acceptance  : {result.acceptance_rate:.3f}")
    print(f"best cut    : {(result.best_state > 0).sum()} / {n} nodes on +1 side")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--n-steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.n, args.n_steps, args.seed)
