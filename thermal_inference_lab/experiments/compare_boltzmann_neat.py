"""Compare normal vs. Boltzmann selection in NEAT maze evolution.

Runs both selection modes with the same seed and prints a side-by-side
comparison of key metrics.

Usage
-----
::

    python -m thermal_inference_lab.experiments.compare_boltzmann_neat [OPTIONS]

Options
-------
--pop          Population size (default 60).
--gens         Number of generations (default 30).
--seed         Random seed (default 42).
--t-start      Boltzmann start temperature (default 5.0).
--t-end        Boltzmann end temperature (default 0.1).
--schedule     Temperature schedule (default "exponential").
--quiet        Suppress per-generation output.
"""

from __future__ import annotations

import argparse
import sys

from thermal_inference_lab.neat.evolution import NeatConfig, evolve
from thermal_inference_lab.neat.maze.environment import BUILT_IN_MAZES
from thermal_inference_lab.neat.results import EvolutionResult


def _print_comparison(normal: EvolutionResult, boltzmann: EvolutionResult) -> None:
    print("\n" + "=" * 64)
    print(f"{'Metric':<30s} {'Normal':>15s} {'Boltzmann':>15s}")
    print("-" * 64)
    print(f"{'Best fitness':<30s} {normal.best_fitness:>15.2f} {boltzmann.best_fitness:>15.2f}")
    print(f"{'Best energy':<30s} {normal.best_energy:>15.2f} {boltzmann.best_energy:>15.2f}")
    print(f"{'Final success rate':<30s} {normal.final_success_rate:>14.0%} {boltzmann.final_success_rate:>14.0%}")

    if normal.history and boltzmann.history:
        n_last = normal.history[-1]
        b_last = boltzmann.history[-1]
        print(f"{'Final species count':<30s} {n_last.num_species:>15d} {b_last.num_species:>15d}")
        print(f"{'Final mean complexity':<30s} {n_last.mean_complexity:>15.1f} {b_last.mean_complexity:>15.1f}")
        print(f"{'Final diversity':<30s} {n_last.diversity:>15.2f} {b_last.diversity:>15.2f}")

    if boltzmann.temperature_history:
        print(f"{'Temperature range':<30s} {'N/A':>15s} "
              f"{boltzmann.temperature_history[0]:.2f}->{boltzmann.temperature_history[-1]:.2f}")
    print("=" * 64)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare normal vs. Boltzmann NEAT selection on mazes.",
    )
    parser.add_argument("--pop", type=int, default=60)
    parser.add_argument("--gens", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t-start", type=float, default=5.0)
    parser.add_argument("--t-end", type=float, default=0.1)
    parser.add_argument("--schedule", type=str, default="exponential",
                        choices=["exponential", "linear"])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    mazes = [fn() for fn in BUILT_IN_MAZES.values()]

    base = dict(
        population_size=args.pop,
        n_generations=args.gens,
        seed=args.seed,
        verbose=not args.quiet,
    )

    print(">>> Normal selection")
    normal_cfg = NeatConfig(selection_mode="normal", **base)
    normal_result = evolve(normal_cfg, mazes)

    print("\n>>> Boltzmann selection")
    boltzmann_cfg = NeatConfig(
        selection_mode="boltzmann",
        t_start=args.t_start,
        t_end=args.t_end,
        schedule_type=args.schedule,
        **base,
    )
    boltzmann_result = evolve(boltzmann_cfg, mazes)

    _print_comparison(normal_result, boltzmann_result)


if __name__ == "__main__":
    main()