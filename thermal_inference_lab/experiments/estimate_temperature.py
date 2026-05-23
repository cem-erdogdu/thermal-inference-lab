"""Recover the temperature at which an Ising sample was drawn.

Generate samples at a known T, hide T, then run
:func:`estimate_temperature` to confirm we recover it.
"""

from __future__ import annotations

import argparse

import numpy as np

from thermal_inference_lab.datasets.synthetic_spins import generate_ising_samples
from thermal_inference_lab.inference.temperature_estimation import estimate_temperature


def main(true_temperature: float = 2.6, size: int = 8, n_samples: int = 400, seed: int = 0) -> None:
    samples = generate_ising_samples(
        shape=(size, size),
        J=1.0,
        h=0.0,
        temperature=true_temperature,
        n_samples=n_samples,
        burn_in=1000,
        interval=4,
        seed=seed,
    )
    fit = estimate_temperature(samples, J=1.0, h=0.0)
    print(f"true T = {true_temperature:.3f}")
    print(f"fit  T = {fit.temperature:.3f}  log_PL = {fit.log_pseudolikelihood:.4f}")
    print(f"rel error = {100 * abs(fit.temperature - true_temperature) / true_temperature:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--true-temperature", type=float, default=2.6)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.true_temperature, args.size, args.n_samples, args.seed)
