"""Gibbs (heat-bath) sampler for the 2D Ising model.

For each site the conditional distribution is

    p(s_i = +1 | rest) = sigmoid( 2 β (J * Σ_nn + h) )

We sample directly from this Bernoulli at every site rather than
proposing a flip — this is "always accept", so the acceptance rate is
trivially 1.0 and is reported only for API symmetry with Metropolis.

This implementation uses a *site-by-site* sweep: we visit every lattice
site once per sweep in a deterministic checkerboard order, which is
provably ergodic and avoids the bias introduced by sampling all sites
in parallel from stale neighbour state.
"""

from __future__ import annotations

import numpy as np

from thermal_inference_lab.core.math_utils import sigmoid
from thermal_inference_lab.energy.ising import IsingModel
from thermal_inference_lab.samplers.base import MCMCSampler


class GibbsSampler(MCMCSampler):
    """Heat-bath Gibbs sampler for the 2D Ising model."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, IsingModel):
            raise TypeError("GibbsSampler currently supports IsingModel only")

    def step(self, state: np.ndarray) -> np.ndarray:
        beta = 1.0 / self.temperature
        rows, cols = state.shape
        # Checkerboard sweep: even sites, then odd sites.
        for parity in (0, 1):
            rs, cs = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
            mask = ((rs + cs) % 2) == parity
            # Compute the local field for *all* sites with current state.
            field = self.model.local_fields(state)
            p_plus = sigmoid(2.0 * beta * field)
            u = self.rng.random(size=state.shape)
            new = np.where(u < p_plus, 1, -1).astype(state.dtype)
            # Only commit the parity-matching subset; the rest stays.
            state = np.where(mask, new, state)
            self.n_proposed += int(mask.sum())
            self.n_accepted += int(mask.sum())  # heat-bath is always-accept
        return state


__all__ = ["GibbsSampler"]
