"""Sample a 2D Ising model at multiple temperatures.

Run from the repo root with::

    python -m thermal_inference_lab.experiments.run_ising

Outputs a figure ``artifacts/ising_magnetization.png`` and prints
summary statistics for each temperature in the ladder.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from thermal_inference_lab.core.config import IsingConfig, SamplerConfig
from thermal_inference_lab.core.constants import ISING_2D_CRITICAL_TEMPERATURE
from thermal_inference_lab.distributions.observables import (
    mean_absolute_magnetization,
    mean_energy,
    specific_heat,
    susceptibility,
)
from thermal_inference_lab.energy.ising import IsingModel
from thermal_inference_lab.samplers.metropolis import MetropolisSampler
from thermal_inference_lab.visualization.phase_plot import (
    plot_magnetization_curve,
    plot_specific_heat_curve,
)


def main(
    size: int = 12,
    n_samples: int = 200,
    burn_in: int = 500,
    interval: int = 2,
    out_dir: str = "artifacts",
    seed: int = 0,
) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    model = IsingModel(IsingConfig(shape=(size, size)))
    temperatures = np.linspace(1.5, 3.5, 9)
    mags, energies_mean, cv_vals, chi_vals = [], [], [], []
    print(f"{'T':>6} {'|m|':>8} {'<E>/N':>10} {'C/N':>8} {'chi/N':>10} {'acc':>6}")
    for t in temperatures:
        sampler = MetropolisSampler(model, temperature=t, rng=seed)
        cfg = SamplerConfig(n_samples=n_samples, burn_in=burn_in, interval=interval, seed=seed)
        result = sampler.run(cfg)
        mags.append(mean_absolute_magnetization(result.samples))
        energies_mean.append(mean_energy(result.energies) / model.n_sites)
        cv_vals.append(specific_heat(result.energies, t, model.n_sites))
        chi_vals.append(susceptibility(result.samples, t) / model.n_sites)
        print(f"{t:6.2f} {mags[-1]:8.4f} {energies_mean[-1]:10.4f} "
              f"{cv_vals[-1]:8.4f} {chi_vals[-1]:10.4f} {result.acceptance_rate:6.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    plot_magnetization_curve(temperatures, np.asarray(mags), ax=axes[0])
    plot_specific_heat_curve(temperatures, np.asarray(cv_vals), ax=axes[1])
    fig.tight_layout()
    out_path = os.path.join(out_dir, "ising_magnetization.png")
    fig.savefig(out_path, dpi=110)
    print(f"\nWrote {out_path}")
    print(f"Onsager T_c (infinite lattice) = {ISING_2D_CRITICAL_TEMPERATURE:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()
    main(
        size=args.size,
        n_samples=args.n_samples,
        burn_in=args.burn_in,
        interval=args.interval,
        out_dir=args.out_dir,
        seed=args.seed,
    )
