"""Classical harmonic oscillator energy.

Provided mainly as a sanity-check target for samplers, simulated
annealing, and the partition-function code paths: the analytical
free energy is known in closed form, so we have ground truth.

    H(x) = (1/2) * k * ||x - x0||^2

We support arbitrary-dimensional x. The model does not implement
single-coordinate "flip" semantics; ``local_energy_delta`` is computed
for an arbitrary proposed perturbation passed in via ``delta_x``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thermal_inference_lab.core.rng import RNG
from thermal_inference_lab.core.validation import require_finite, require_positive


@dataclass(frozen=True)
class HarmonicConfig:
    n_dim: int = 1
    k: float = 1.0
    x0: float = 0.0
    proposal_scale: float = 0.5


class HarmonicOscillator:
    """Continuous-state harmonic oscillator (not an EnergyModel subclass).

    We do not derive from :class:`EnergyModel` because that interface
    assumes single-site flips, which is meaningless for continuous
    state. Instead this class exposes the methods samplers actually
    need: ``total_energy`` and ``propose``.
    """

    def __init__(self, config: HarmonicConfig | None = None, **overrides) -> None:
        if config is None:
            config = HarmonicConfig(**overrides)
        require_positive(config.k, name="k")
        require_positive(config.proposal_scale, name="proposal_scale")
        self.config = config

    @property
    def n_dim(self) -> int:
        return self.config.n_dim

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self.n_dim,)

    def total_energy(self, x: np.ndarray) -> float:
        require_finite(x, name="x")
        diff = x.astype(np.float64) - self.config.x0
        return float(0.5 * self.config.k * (diff @ diff))

    def random_state(self, rng: RNG) -> np.ndarray:
        return rng.normal(self.config.x0, scale=1.0, size=self.n_dim)

    def propose(self, x: np.ndarray, rng: RNG) -> np.ndarray:
        """Symmetric Gaussian random-walk proposal."""
        return x + rng.normal(0.0, self.config.proposal_scale, size=x.shape)

    def analytic_log_partition(self, temperature: float) -> float:
        """log Z = (n_dim / 2) * log(2 pi T / k) for T > 0."""
        require_positive(temperature, name="temperature")
        return float(
            0.5 * self.n_dim * np.log(2.0 * np.pi * temperature / self.config.k)
        )

    def analytic_mean_energy(self, temperature: float) -> float:
        """<E> = (n_dim / 2) * T  (equipartition)."""
        require_positive(temperature, name="temperature")
        return float(0.5 * self.n_dim * temperature)


__all__ = ["HarmonicOscillator", "HarmonicConfig"]
