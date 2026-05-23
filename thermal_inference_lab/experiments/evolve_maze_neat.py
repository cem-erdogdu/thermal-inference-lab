"""Evolve NEAT agents to solve grid mazes.

Usage
-----
::

    python -m thermal_inference_lab.experiments.evolve_maze_neat [OPTIONS]

Options
-------
--pop          Population size (default 80).
--gens         Number of generations (default 40).
--mode         Selection mode: "normal" or "boltzmann" (default "normal").
--seed         Random seed (default 42).
--t-start      Boltzmann start temperature (default 5.0).
--t-end        Boltzmann end temperature (default 0.1).
--schedule     Temperature schedule: "exponential" or "linear" (default "exponential").
--quiet        Suppress per-generation output.
"""

from __future__ import annotations

import argparse
import sys

from thermal_inference_lab.neat.evolution import NeatConfig, evolve
from thermal_inference_lab.neat.maze.environment import BUILT_IN_MAZES


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evolve NEAT agents on grid mazes.",
    )
    parser.add_argument("--pop", type=int, default=80, help="Population size")
    parser.add_argument("--gens", type=int, default=40, help="Number of generations")
    parser.add_argument("--mode", type=str, default="normal",
                        choices=["normal", "boltzmann"], help="Selection mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--t-start", type=float, default=5.0, help="Boltzmann T start")
    parser.add_argument("--t-end", type=float, default=0.1, help="Boltzmann T end")
    parser.add_argument("--schedule", type=str, default="exponential",
                        choices=["exponential", "linear"], help="Temperature schedule")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-gen output")
    args = parser.parse_args(argv)

    mazes = [fn() for fn in BUILT_IN_MAZES.values()]

    config = NeatConfig(
        population_size=args.pop,
        n_generations=args.gens,
        selection_mode=args.mode,
        seed=args.seed,
        t_start=args.t_start,
        t_end=args.t_end,
        schedule_type=args.schedule,
        verbose=not args.quiet,
    )

    print(f"Running NEAT ({args.mode} selection) | pop={args.pop} gens={args.gens} seed={args.seed}")
    if args.mode == "boltzmann":
        print(f"  Temperature: {args.t_start} -> {args.t_end} ({args.schedule})")
    print()

    result = evolve(config, mazes)

    print()
    print(result.summary())

    # Show best path on each maze.
    if result.best_maze_results:
        print("\n--- Best Agent Maze Results ---")
        for name, run in result.best_maze_results.items():
            status = "SOLVED" if run.reached_goal else "FAILED"
            print(f"  {name:12s}: {status}  steps={run.steps_taken}  "
                  f"walls={run.wall_hits}  score={run.score:.1f}")


if __name__ == "__main__":
    main()