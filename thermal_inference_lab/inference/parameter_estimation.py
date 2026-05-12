"""Maximum-pseudolikelihood estimation of Ising parameters (J, h).

Given a batch of equilibrium samples at known inverse temperature
``beta``, fit J and h that maximise the pseudolikelihood. Optionally
estimate beta too: in that case we fix one of (J, h) as the scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from thermal_inference_lab.core.exceptions import InferenceError
from thermal_inference_lab.core.validation import require_positive
from thermal_inference_lab.inference.fit_result import FitResult
from thermal_inference_lab.inference.pseudolikelihood import (
    ising_pseudolikelihood,
    ising_pseudolikelihood_gradient,
)
from thermal_inference_lab.optimizers.adam import Adam
from thermal_inference_lab.optimizers.base import Optimizer


@dataclass
class IsingParameterFit:
    J: float
    h: float
    loss_history: np.ndarray
    converged: bool
    n_iterations: int

    def to_fit_result(self) -> FitResult:
        return FitResult(
            params={"J": self.J, "h": self.h},
            loss_history=self.loss_history,
            converged=self.converged,
            n_iterations=self.n_iterations,
            metadata={"method": "pseudolikelihood"},
        )


def estimate_ising_parameters(
    samples: np.ndarray,
    beta: float = 1.0,
    init_J: float = 0.0,
    init_h: float = 0.0,
    optimizer: Optional[Optimizer] = None,
    n_iter: int = 500,
    tol: float = 1e-6,
    periodic: bool = True,
    verbose: bool = False,
) -> IsingParameterFit:
    """Fit (J, h) by maximum pseudolikelihood with gradient ascent.

    We work in *parameter-dict* form so the same optimizers used for
    the RBM also work here. The loss we minimise is the *negative*
    mean log-pseudolikelihood.
    """
    require_positive(beta, name="beta")
    if optimizer is None:
        optimizer = Adam(lr=0.05)

    params = {"J": np.asarray([float(init_J)]), "h": np.asarray([float(init_h)])}
    history = np.empty(n_iter, dtype=np.float64)
    converged = False
    prev_loss = None
    n_run = 0
    for it in range(n_iter):
        loss = -ising_pseudolikelihood(samples, float(params["J"][0]), float(params["h"][0]), beta=beta, periodic=periodic)
        grads_scalar = ising_pseudolikelihood_gradient(
            samples,
            float(params["J"][0]),
            float(params["h"][0]),
            beta=beta,
            periodic=periodic,
        )
        grads = {
            "J": np.asarray([grads_scalar["J"]]),
            "h": np.asarray([grads_scalar["h"]]),
        }
        optimizer.step(params, grads)
        history[it] = loss
        n_run = it + 1
        if verbose and (it % max(n_iter // 20, 1) == 0):
            print(f"[iter {it:5d}] loss={loss:.6f} J={params['J'][0]:.4f} h={params['h'][0]:.4f}")
        if prev_loss is not None and abs(prev_loss - loss) < tol:
            converged = True
            break
        prev_loss = loss

    return IsingParameterFit(
        J=float(params["J"][0]),
        h=float(params["h"][0]),
        loss_history=history[:n_run],
        converged=converged,
        n_iterations=n_run,
    )


__all__ = ["IsingParameterFit", "estimate_ising_parameters"]
