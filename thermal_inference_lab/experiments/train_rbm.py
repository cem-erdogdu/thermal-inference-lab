"""Train a small RBM on bars-and-stripes using contrastive divergence."""

from __future__ import annotations

import argparse

import numpy as np

from thermal_inference_lab.core.rng import RNG
from thermal_inference_lab.datasets.rbm_patterns import bars_and_stripes
from thermal_inference_lab.energy.rbm import RBMEnergy
from thermal_inference_lab.inference.contrastive_divergence import contrastive_divergence
from thermal_inference_lab.optimizers.adam import Adam


def main(size: int = 4, n_hidden: int = 6, n_epochs: int = 60, seed: int = 0) -> None:
    rng = RNG(seed)
    data = bars_and_stripes(size=size).astype(np.int8)
    print(f"Training RBM on {data.shape[0]} bars-and-stripes patterns ({size}x{size}).")
    rbm = RBMEnergy(n_visible=size * size, n_hidden=n_hidden, rng=rng, weight_scale=0.05)
    fit = contrastive_divergence(
        rbm,
        data=data,
        n_epochs=n_epochs,
        batch_size=8,
        k=1,
        optimizer=Adam(lr=5e-2),
        rng=rng,
        verbose=False,
    )
    print(f"Final reconstruction MSE: {fit.final_loss:.4f}")
    fe = np.array([rbm.free_energy(v) for v in data])
    print(f"Mean F(v) over training set: {fe.mean():.4f}")
    print(f"Min  F(v) over training set: {fe.min():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--n-hidden", type=int, default=6)
    parser.add_argument("--n-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.size, args.n_hidden, args.n_epochs, args.seed)
