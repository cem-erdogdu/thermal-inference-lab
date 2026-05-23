"""High-level orchestration for running multiple chains/temperatures.

The :class:`Chain` helper is a thin wrapper around :class:`MCMCSampler`
that records the trajectory at each step (not only every ``interval``
samples) and lets you replay an entire history. Useful for visualising
sampler dynamics in experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from thermal_inference_lab.core.config import SamplerConfig
from thermal_inference_lab.core.rng import RNG, as_rng
from thermal_inference_lab.samplers.base import MCMCSampler, SamplerResult


@dataclass
class ChainHistory:
    states: np.ndarray  # (n_steps, ...)
    energies: np.ndarray  # (n_steps,)
    acceptance_rate: float
    sampler_kind: str

    @property
    def n_steps(self) -> int:
        return self.states.shape[0]


class Chain:
    """Wrapper that runs a sampler and records the *full* trajectory."""

    def __init__(self, sampler: MCMCSampler, name: str = "chain") -> None:
        self.sampler = sampler
        self.name = name

    def run(
        self,
        n_steps: int,
        burn_in: int = 0,
        initial_state: Optional[np.ndarray] = None,
        seed: int = 0,
    ) -> ChainHistory:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        self.sampler.rng = RNG(seed)
        self.sampler.n_proposed = 0
        self.sampler.n_accepted = 0

        state = (
            initial_state.copy()
            if initial_state is not None
            else self.sampler.model.random_state(self.sampler.rng)
        )
        for _ in range(burn_in):
            state = self.sampler.step(state)
        self.sampler.n_proposed = 0
        self.sampler.n_accepted = 0

        states = np.empty((n_steps,) + state.shape, dtype=state.dtype)
        energies = np.empty(n_steps, dtype=np.float64)
        for i in range(n_steps):
            state = self.sampler.step(state)
            states[i] = state
            energies[i] = self.sampler.model.total_energy(state)

        rate = (
            self.sampler.n_accepted / self.sampler.n_proposed
            if self.sampler.n_proposed > 0
            else 0.0
        )
        return ChainHistory(
            states=states,
            energies=energies,
            acceptance_rate=float(rate),
            sampler_kind=type(self.sampler).__name__,
        )


def run_temperature_ladder(
    sampler_factory,
    temperatures,
    base_config: SamplerConfig,
) -> dict[float, SamplerResult]:
    """Run one chain per temperature, returning a dict keyed by T."""
    out: dict[float, SamplerResult] = {}
    for t in temperatures:
        sampler = sampler_factory(t)
        out[float(t)] = sampler.run(base_config)
    return out


__all__ = ["Chain", "ChainHistory", "run_temperature_ladder"]
