"""Single-spin-flip Metropolis sampler.

For each lattice site visited, propose a single-site flip, compute the
local energy delta, and accept with probability ``min(1, exp(-βΔE))``.
One *sweep* consists of ``n_sites`` such updates (sites chosen uniformly
at random with replacement — the standard Metropolis recipe).

The sampler is generic over spin/occupancy semantics: it asks the
energy model for the local delta and uses :meth:`EnergyModel.copy_with_flip`
to mutate the state. Ising models override ``copy_with_flip`` to do a
sign flip; lattice-gas / defect models flip 0 <-> 1.
"""

from __future__ import annotations

import numpy as np

from thermal_inference_lab.distributions.boltzmann import boltzmann_acceptance
from thermal_inference_lab.samplers.base import MCMCSampler


class MetropolisSampler(MCMCSampler):
    """Random-site Metropolis sampler for 2D lattice models."""

    def __init__(self, *args, in_place_flip: bool | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # ``in_place_flip`` defaults to True for Ising (±1), False otherwise.
        if in_place_flip is None:
            from thermal_inference_lab.energy.ising import IsingModel
            in_place_flip = isinstance(self.model, IsingModel)
        self._in_place_flip = in_place_flip

    def step(self, state: np.ndarray) -> np.ndarray:
        if state.ndim != 2:
            raise NotImplementedError(
                "MetropolisSampler currently expects 2D states; "
                "use a custom subclass for other geometries"
            )
        rows, cols = state.shape
        n_sites = rows * cols
        site_rows = self.rng.integers(0, rows, size=n_sites)
        site_cols = self.rng.integers(0, cols, size=n_sites)
        uniforms = self.rng.random(size=n_sites)
        for k in range(n_sites):
            r, c = int(site_rows[k]), int(site_cols[k])
            d_e = self.model.local_energy_delta(state, (r, c))
            self.n_proposed += 1
            if d_e <= 0.0 or uniforms[k] < boltzmann_acceptance(d_e, self.temperature):
                if self._in_place_flip:
                    state[r, c] = -state[r, c]
                else:
                    state[r, c] = 1 - state[r, c]
                self.n_accepted += 1
        return state


__all__ = ["MetropolisSampler"]
