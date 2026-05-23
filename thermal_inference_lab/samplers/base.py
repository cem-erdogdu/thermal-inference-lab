"""Common scaffolding for MCMC samplers.

All concrete samplers subclass :class:`MCMCSampler` and implement a
single ``step()`` method that advances the chain by one sweep (lattice
sized). The base class is responsible for:

* burn-in
* thinning by ``interval``
* storing the energy trace
* recording acceptance diagnostics (where applicable)
* exposing a reproducible :class:`RNG` to subclasses
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from thermal_inference_lab.core.config import SamplerConfig
from thermal_inference_lab.core.rng import RNG, as_rng
from thermal_inference_lab.core.validation import require_positive
from thermal_inference_lab.energy.base import EnergyModel


@dataclass
class SamplerResult:
    """Container returned by :meth:`MCMCSampler.run`."""

    samples: np.ndarray
    energies: np.ndarray
    acceptance_rate: float
    final_state: np.ndarray
    n_proposed: int
    n_accepted: int
    config: SamplerConfig

    @property
    def n_samples(self) -> int:
        return self.samples.shape[0]


class MCMCSampler(ABC):
    """Base class for Metropolis-style single-spin samplers."""

    def __init__(
        self,
        model: EnergyModel,
        temperature: float,
        rng: RNG | int | None = None,
    ) -> None:
        require_positive(temperature, name="temperature")
        self.model = model
        self.temperature = float(temperature)
        self.rng: RNG = as_rng(rng)
        self.n_proposed: int = 0
        self.n_accepted: int = 0

    # ------------------------------------------------------------ step API
    @abstractmethod
    def step(self, state: np.ndarray) -> np.ndarray:
        """Advance the chain by one full *sweep* (n_sites single-site updates).

        Implementations must update ``self.n_proposed`` and
        ``self.n_accepted`` for diagnostics.
        """

    # ----------------------------------------------------- driver methods
    def run(
        self,
        config: SamplerConfig | None = None,
        initial_state: Optional[np.ndarray] = None,
        **overrides,
    ) -> SamplerResult:
        if config is None:
            config = SamplerConfig(**overrides)
        elif overrides:
            raise ValueError("pass config or kwargs, not both")
        # Re-seed so that the same SamplerConfig.seed reproduces results.
        self.rng = RNG(config.seed)
        self.n_proposed = 0
        self.n_accepted = 0

        state = (
            initial_state.copy()
            if initial_state is not None
            else self.model.random_state(self.rng)
        )

        # Burn-in.
        for _ in range(config.burn_in):
            state = self.step(state)
        # Reset acceptance counters AFTER burn-in so the reported rate
        # reflects the production phase, not warm-up.
        self.n_proposed = 0
        self.n_accepted = 0

        samples = np.empty((config.n_samples,) + state.shape, dtype=state.dtype)
        energies = np.empty(config.n_samples, dtype=np.float64) if config.record_energy else None

        for i in range(config.n_samples):
            for _ in range(config.interval):
                state = self.step(state)
            samples[i] = state
            if energies is not None:
                energies[i] = self.model.total_energy(state)

        rate = (self.n_accepted / self.n_proposed) if self.n_proposed > 0 else 0.0
        return SamplerResult(
            samples=samples,
            energies=energies if energies is not None else np.empty(0),
            acceptance_rate=float(rate),
            final_state=state.copy(),
            n_proposed=self.n_proposed,
            n_accepted=self.n_accepted,
            config=config,
        )


__all__ = ["MCMCSampler", "SamplerResult"]
