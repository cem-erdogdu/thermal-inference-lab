"""Synthetic spin datasets for tests and experiments."""

from __future__ import annotations

import numpy as np

from thermal_inference_lab.core.config import IsingConfig, SamplerConfig
from thermal_inference_lab.core.rng import RNG, as_rng
from thermal_inference_lab.energy.ising import IsingModel
from thermal_inference_lab.samplers.metropolis import MetropolisSampler


def generate_ising_samples(
    shape: tuple[int, int] = (8, 8),
    J: float = 1.0,
    h: float = 0.0,
    temperature: float = 2.5,
    n_samples: int = 200,
    burn_in: int = 500,
    interval: int = 5,
    seed: int = 0,
    periodic: bool = True,
) -> np.ndarray:
    """Return ``n_samples`` equilibrium configs at the given temperature."""
    model = IsingModel(IsingConfig(shape=shape, J=J, h=h, periodic=periodic))
    sampler = MetropolisSampler(model, temperature=temperature, rng=seed)
    result = sampler.run(
        SamplerConfig(
            n_samples=n_samples,
            burn_in=burn_in,
            interval=interval,
            seed=seed,
            record_energy=False,
        )
    )
    return result.samples


def generate_correlated_spins(
    n_samples: int,
    n_spins: int,
    correlation: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Cheap dataset of {-1, +1} vectors with a tunable shared latent bias.

    Each sample shares a latent ``z ~ Uniform({-1,+1})`` with probability
    ``(1 + correlation) / 2``, and otherwise is independent. Useful for
    smoke-testing inference without needing a full MCMC run.
    """
    rng = as_rng(seed)
    z = rng.integers(0, 2, size=n_samples).astype(np.int8) * 2 - 1
    flip_prob = 0.5 * (1.0 - correlation)
    base = np.repeat(z[:, None], n_spins, axis=1)
    flips = rng.random(base.shape) < flip_prob
    return np.where(flips, -base, base).astype(np.int8)


__all__ = ["generate_ising_samples", "generate_correlated_spins"]
