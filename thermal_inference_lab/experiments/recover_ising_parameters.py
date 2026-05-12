"""Recover (J, h) from Ising samples via maximum pseudolikelihood."""

from __future__ import annotations

import argparse

from thermal_inference_lab.datasets.synthetic_spins import generate_ising_samples
from thermal_inference_lab.inference.parameter_estimation import estimate_ising_parameters


def main(
    true_J: float = 1.0,
    true_h: float = 0.2,
    size: int = 8,
    temperature: float = 2.5,
    n_samples: int = 400,
    seed: int = 0,
) -> None:
    samples = generate_ising_samples(
        shape=(size, size),
        J=true_J,
        h=true_h,
        temperature=temperature,
        n_samples=n_samples,
        burn_in=1000,
        interval=4,
        seed=seed,
    )
    fit = estimate_ising_parameters(
        samples,
        beta=1.0 / temperature,
        init_J=0.0,
        init_h=0.0,
        n_iter=600,
        tol=1e-7,
    )
    print(f"true (J, h)     = ({true_J:.4f}, {true_h:.4f})")
    print(f"recovered (J, h) = ({fit.J:.4f}, {fit.h:.4f})")
    print(f"converged={fit.converged} after {fit.n_iterations} iterations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--true-J", type=float, default=1.0)
    parser.add_argument("--true-h", type=float, default=0.2)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=2.5)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.true_J, args.true_h, args.size, args.temperature, args.n_samples, args.seed)
