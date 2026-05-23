"""Compare Metropolis vs Gibbs samplers on the same Ising model."""

from __future__ import annotations

import argparse

from thermal_inference_lab.core.config import IsingConfig, SamplerConfig
from thermal_inference_lab.distributions.observables import mean_absolute_magnetization, mean_energy
from thermal_inference_lab.energy.ising import IsingModel
from thermal_inference_lab.samplers.gibbs import GibbsSampler
from thermal_inference_lab.samplers.metropolis import MetropolisSampler
from thermal_inference_lab.samplers.diagnostics import summarize_chain


def main(size: int = 10, temperature: float = 2.27, n_samples: int = 400, seed: int = 0) -> None:
    model = IsingModel(IsingConfig(shape=(size, size)))
    cfg = SamplerConfig(n_samples=n_samples, burn_in=500, interval=2, seed=seed)

    for name, sampler in [
        ("metropolis", MetropolisSampler(model, temperature=temperature, rng=seed)),
        ("gibbs", GibbsSampler(model, temperature=temperature, rng=seed)),
    ]:
        result = sampler.run(cfg)
        diag = summarize_chain(result)
        print(
            f"[{name:10s}] <E>={diag.mean_energy: .4f}  |m|={mean_absolute_magnetization(result.samples): .4f}"
            f"  acc={result.acceptance_rate: .3f}  ESS={diag.effective_sample_size:8.1f}  tau_int={diag.tau_int:6.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=2.27)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.size, args.temperature, args.n_samples, args.seed)
