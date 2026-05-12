import numpy as np
import pytest

from thermal_inference_lab.core.rng import RNG
from thermal_inference_lab.datasets.rbm_patterns import bars_and_stripes
from thermal_inference_lab.energy.rbm import RBMEnergy
from thermal_inference_lab.inference.contrastive_divergence import contrastive_divergence
from thermal_inference_lab.optimizers.adam import Adam


def test_cd_decreases_reconstruction_error():
    """CD-1 should reduce reconstruction MSE on the training data."""
    rng = RNG(0)
    data = bars_and_stripes(size=4).astype(np.int8)
    rbm = RBMEnergy(n_visible=16, n_hidden=8, rng=rng, weight_scale=0.05)
    fit = contrastive_divergence(
        rbm,
        data=data,
        n_epochs=30,
        batch_size=4,
        k=1,
        optimizer=Adam(lr=1e-2),
        rng=rng,
    )
    # Average loss in the last quarter should be below the first quarter.
    n = fit.loss_history.size
    first = fit.loss_history[: n // 4].mean()
    last = fit.loss_history[-n // 4 :].mean()
    assert last < first


def test_cd_returns_loss_history():
    rng = RNG(0)
    data = bars_and_stripes(size=3).astype(np.int8)
    rbm = RBMEnergy(n_visible=9, n_hidden=4, rng=rng, weight_scale=0.05)
    fit = contrastive_divergence(
        rbm, data=data, n_epochs=5, batch_size=4, optimizer=Adam(lr=1e-2), rng=rng
    )
    assert fit.loss_history.size == 5
    assert np.all(np.isfinite(fit.loss_history))
