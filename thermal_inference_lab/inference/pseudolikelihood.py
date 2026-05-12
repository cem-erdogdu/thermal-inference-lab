"""Pseudolikelihood for the 2D Ising model.

The full likelihood of a sample x under the Ising model is intractable
because of the partition function. Pseudolikelihood (Besag 1975)
approximates it with a product of one-site conditionals:

    log PL(x; J, h, beta) = sum_i log p(x_i | x_{-i}; J, h, beta)

For Ising this becomes

    log p(x_i = +1 | x_{-i}) = -log(1 + exp(-2 * beta * (J * sum_nn + h)))
    log p(x_i = -1 | x_{-i}) = -log(1 + exp(+2 * beta * (J * sum_nn + h)))

which is the binary cross-entropy of the centred field. The closed-form
gradient is implemented to enable fitting with the package's NumPy
optimizers.

All functions take a *batch* of configurations of shape (B, R, C).
"""

from __future__ import annotations

import numpy as np

from thermal_inference_lab.core.math_utils import softplus, sigmoid
from thermal_inference_lab.core.validation import require_positive, require_spin_array


def _neighbour_sum_batch(samples: np.ndarray, periodic: bool = True) -> np.ndarray:
    """Vectorised 4-neighbour sum over (B, R, C) batched configs."""
    if periodic:
        return (
            np.roll(samples, 1, axis=1)
            + np.roll(samples, -1, axis=1)
            + np.roll(samples, 1, axis=2)
            + np.roll(samples, -1, axis=2)
        )
    padded = np.pad(samples, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0)
    return (
        padded[:, :-2, 1:-1]
        + padded[:, 2:, 1:-1]
        + padded[:, 1:-1, :-2]
        + padded[:, 1:-1, 2:]
    )


def ising_pseudolikelihood(
    samples: np.ndarray,
    J: float,
    h: float,
    beta: float = 1.0,
    periodic: bool = True,
) -> float:
    """Mean log-pseudolikelihood per site, averaged over the batch.

    Higher is better; this matches the convention used as the inference
    *objective* (we maximise it).
    """
    require_spin_array(samples[0], ndim=2)
    require_positive(beta, name="beta")
    sums = _neighbour_sum_batch(samples, periodic=periodic).astype(np.float64)
    s = samples.astype(np.float64)
    field = J * sums + h
    # log p(s_i | rest) = - softplus(-2 * beta * s_i * field)
    log_p = -softplus(-2.0 * beta * s * field)
    return float(log_p.mean())


def ising_pseudolikelihood_gradient(
    samples: np.ndarray,
    J: float,
    h: float,
    beta: float = 1.0,
    periodic: bool = True,
) -> dict[str, float]:
    """Gradient of the *negative* mean log-PL w.r.t. (J, h).

    We return the gradient of the *loss* (negative PL) so it can be
    passed directly to a minimising optimizer such as SGD or Adam.
    """
    require_positive(beta, name="beta")
    sums = _neighbour_sum_batch(samples, periodic=periodic).astype(np.float64)
    s = samples.astype(np.float64)
    field = J * sums + h
    z = -2.0 * beta * s * field
    # d/dz softplus(z) = sigmoid(z)
    # d/dz of -softplus(z) = -sigmoid(z)
    # d/dx of -softplus(-2 beta s x) = 2 beta s sigmoid(-2 beta s field) wrt field
    coeff = 2.0 * beta * s * sigmoid(z)  # element-wise (B, R, C)
    # Gradient of log PL averaged over batch.
    grad_J = coeff * sums
    grad_h = coeff
    # Negate because the optimizer minimises the loss.
    return {
        "J": float(-grad_J.mean()),
        "h": float(-grad_h.mean()),
    }


__all__ = [
    "ising_pseudolikelihood",
    "ising_pseudolikelihood_gradient",
]
