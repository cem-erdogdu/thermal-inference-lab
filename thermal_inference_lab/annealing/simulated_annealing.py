"""Simulated annealing driver for combinatorial spin/occupancy problems.

We use single-site flips and the Metropolis acceptance rule with a
shrinking temperature dictated by a :class:`Schedule`. The driver
tracks the best (lowest-energy) state seen so far, so the run is
guaranteed to return a configuration no worse than the one it
encountered, regardless of late-stage rejections.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thermal_inference_lab.annealing.acceptance import metropolis_accept
from thermal_inference_lab.annealing.schedules import Schedule, schedule_from_name
from thermal_inference_lab.core.config import AnnealConfig
from thermal_inference_lab.core.rng import RNG, as_rng
from thermal_inference_lab.energy.base import EnergyModel


@dataclass
class AnnealingResult:
    best_state: np.ndarray
    best_energy: float
    final_state: np.ndarray
    final_energy: float
    energy_trace: np.ndarray
    temperature_trace: np.ndarray
    n_accepted: int
    n_proposed: int

    @property
    def acceptance_rate(self) -> float:
        return self.n_accepted / max(self.n_proposed, 1)


class SimulatedAnnealing:
    """Single-flip simulated annealing for generic :class:`EnergyModel`."""

    def __init__(
        self,
        model: EnergyModel,
        schedule: Schedule | None = None,
        rng: RNG | int | None = None,
    ) -> None:
        self.model = model
        self.schedule = schedule
        self.rng = as_rng(rng)

    @classmethod
    def from_config(cls, model: EnergyModel, config: AnnealConfig) -> "SimulatedAnnealing":
        schedule = schedule_from_name(
            config.schedule, config.t_start, config.t_end, config.n_steps
        )
        return cls(model=model, schedule=schedule, rng=config.seed)

    def run(
        self,
        initial_state: np.ndarray | None = None,
        n_steps: int | None = None,
    ) -> AnnealingResult:
        if self.schedule is None:
            raise ValueError("schedule must be set before run()")
        if n_steps is None:
            n_steps = self.schedule.n_steps
        state = (
            initial_state.copy()
            if initial_state is not None
            else self.model.random_state(self.rng)
        )

        in_place_flip = self._uses_sign_flip(state)
        current_e = self.model.total_energy(state)
        best_state = state.copy()
        best_e = current_e
        energies = np.empty(n_steps, dtype=np.float64)
        temps = np.empty(n_steps, dtype=np.float64)

        n_proposed = 0
        n_accepted = 0
        shape = state.shape
        for step in range(n_steps):
            t = self.schedule(step)
            temps[step] = t
            # Propose a flip at a random site.
            idx = tuple(int(self.rng.integers(0, s)) for s in shape)
            d_e = self.model.local_energy_delta(state, idx)
            u = float(self.rng.uniform(0.0, 1.0))
            n_proposed += 1
            if metropolis_accept(d_e, t, u):
                if in_place_flip:
                    state[idx] = -state[idx]
                else:
                    state[idx] = 1 - state[idx]
                current_e += d_e
                n_accepted += 1
                if current_e < best_e:
                    best_e = current_e
                    best_state = state.copy()
            energies[step] = current_e

        return AnnealingResult(
            best_state=best_state,
            best_energy=float(best_e),
            final_state=state.copy(),
            final_energy=float(current_e),
            energy_trace=energies,
            temperature_trace=temps,
            n_accepted=n_accepted,
            n_proposed=n_proposed,
        )

    @staticmethod
    def _uses_sign_flip(state: np.ndarray) -> bool:
        return bool(np.any(state == -1))


__all__ = ["SimulatedAnnealing", "AnnealingResult"]
