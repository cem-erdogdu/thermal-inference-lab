"""Contrastive Divergence (CD-k) training for the RBM.

The negative log-likelihood gradient under the RBM model is

    ∂NLL/∂θ  =  <∂F/∂θ>_data  -  <∂F/∂θ>_model

CD-k approximates the second expectation with k steps of block Gibbs
sampling started from the data. We implement CD-1 by default; higher
k is supported via the ``k`` argument.
"""

from __future__ import annotations

import numpy as np

from thermal_inference_lab.core.rng import RNG, as_rng
from thermal_inference_lab.core.validation import require_positive
from thermal_inference_lab.energy.rbm import RBMEnergy
from thermal_inference_lab.inference.fit_result import FitResult
from thermal_inference_lab.optimizers.adam import Adam
from thermal_inference_lab.optimizers.base import Optimizer


def _gibbs_step(rbm: RBMEnergy, v: np.ndarray, rng: RNG) -> np.ndarray:
    """One full Gibbs step v -> h -> v over a batch."""
    h = rbm.sample_hidden_given_visible(v, rng)
    v_new = rbm.sample_visible_given_hidden(h, rng)
    return v_new


def contrastive_divergence(
    rbm: RBMEnergy,
    data: np.ndarray,
    n_epochs: int = 20,
    batch_size: int = 32,
    k: int = 1,
    optimizer: Optimizer | None = None,
    rng: RNG | int | None = None,
    verbose: bool = False,
) -> FitResult:
    """Train an RBM with CD-k. Mutates ``rbm.params`` in place."""
    if data.ndim != 2 or data.shape[1] != rbm.n_visible:
        raise ValueError(
            f"data shape {data.shape} incompatible with rbm.n_visible={rbm.n_visible}"
        )
    require_positive(n_epochs, name="n_epochs")
    require_positive(batch_size, name="batch_size")
    require_positive(k, name="k")
    rng = as_rng(rng)
    if optimizer is None:
        optimizer = Adam(lr=1e-2)

    n = data.shape[0]
    losses = []

    for epoch in range(n_epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch = data[order[start : start + batch_size]].astype(np.float64)
            # Negative phase: k Gibbs steps starting from data.
            neg = batch.copy()
            for _ in range(k):
                neg = _gibbs_step(rbm, neg, rng)
            grad_pos = rbm.free_energy_gradients(batch)
            grad_neg = rbm.free_energy_gradients(neg)
            grads = {
                k_: grad_pos[k_] - grad_neg[k_] for k_ in grad_pos
            }
            params = rbm.params.as_dict()
            optimizer.step(params, grads)
            # Track reconstruction MSE as a cheap proxy for likelihood.
            recon = rbm.prob_visible_given_hidden(
                rbm.prob_hidden_given_visible(batch)
            )
            epoch_loss += float(np.mean((batch - recon) ** 2))
            n_batches += 1
        epoch_loss /= max(n_batches, 1)
        losses.append(epoch_loss)
        if verbose:
            print(f"[epoch {epoch:3d}] recon_mse={epoch_loss:.5f}")

    return FitResult(
        params={"W": rbm.params.W.copy(), "a": rbm.params.a.copy(), "b": rbm.params.b.copy()},
        loss_history=np.asarray(losses, dtype=np.float64),
        converged=True,
        n_iterations=n_epochs,
        metadata={"method": f"CD-{k}"},
    )


__all__ = ["contrastive_divergence"]
