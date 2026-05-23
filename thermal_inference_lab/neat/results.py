"""Structured result objects and diagnostics for NEAT evolution runs.

:class:`GenerationSnapshot` captures one generation's statistics.
:class:`EvolutionResult` wraps the full history and provides summary
accessors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from thermal_inference_lab.neat.genome import Genome
from thermal_inference_lab.neat.maze.evaluation import MazeRunResult


@dataclass
class GenerationSnapshot:
    """Statistics for a single generation."""
    generation: int
    best_fitness: float
    mean_fitness: float
    best_energy: float
    mean_energy: float
    success_rate: float  # fraction of genomes that solved all mazes
    num_species: int
    population_size: int
    mean_complexity: float
    max_complexity: float
    diversity: float  # fraction of unique fitness values
    temperature: Optional[float] = None
    best_genome_id: int = -1
    best_path: Optional[List[Tuple[int, int]]] = None

    def summary_line(self) -> str:
        t_str = f"  T={self.temperature:.3f}" if self.temperature is not None else ""
        return (
            f"Gen {self.generation:4d} | "
            f"best={self.best_fitness:7.2f}  mean={self.mean_fitness:7.2f} | "
            f"E_best={self.best_energy:8.2f} | "
            f"succ={self.success_rate:.0%} | "
            f"sp={self.num_species} | "
            f"cx={self.mean_complexity:.1f}{t_str}"
        )


@dataclass
class EvolutionResult:
    """Complete result of a NEAT evolution run."""
    selection_mode: str  # "normal" or "boltzmann"
    generations_run: int = 0
    history: List[GenerationSnapshot] = field(default_factory=list)
    best_genome: Optional[Genome] = None
    best_fitness: float = 0.0
    best_energy: float = 0.0
    best_maze_results: Optional[Dict[str, MazeRunResult]] = None
    temperature_history: List[float] = field(default_factory=list)

    # ----------------------------------------------------- aggregations

    @property
    def fitness_curve(self) -> np.ndarray:
        return np.array([s.best_fitness for s in self.history])

    @property
    def energy_curve(self) -> np.ndarray:
        return np.array([s.best_energy for s in self.history])

    @property
    def species_count_curve(self) -> np.ndarray:
        return np.array([s.num_species for s in self.history])

    @property
    def complexity_curve(self) -> np.ndarray:
        return np.array([s.mean_complexity for s in self.history])

    @property
    def success_rate_curve(self) -> np.ndarray:
        return np.array([s.success_rate for s in self.history])

    @property
    def diversity_curve(self) -> np.ndarray:
        return np.array([s.diversity for s in self.history])

    @property
    def final_success_rate(self) -> float:
        return self.history[-1].success_rate if self.history else 0.0

    # ---------------------------------------------------- printing

    def summary(self) -> str:
        lines = [
            f"=== Evolution Result ({self.selection_mode}) ===",
            f"Generations:       {self.generations_run}",
            f"Best fitness:      {self.best_fitness:.4f}",
            f"Best energy:       {self.best_energy:.4f}",
            f"Final success:     {self.final_success_rate:.0%}",
        ]
        if self.history:
            last = self.history[-1]
            lines.append(f"Final species:     {last.num_species}")
            lines.append(f"Mean complexity:   {last.mean_complexity:.1f}")
            lines.append(f"Population div:    {last.diversity:.2f}")
        if self.best_genome is not None:
            lines.append(f"Best genome:       {self.best_genome}")
        if self.temperature_history:
            lines.append(f"T range:           {self.temperature_history[0]:.3f} -> {self.temperature_history[-1]:.3f}")
        return "\n".join(lines)


def compute_snapshot(
    generation: int,
    population: List[Genome],
    num_species: int,
    maze_results: Dict[int, List[MazeRunResult]],
    temperature: Optional[float] = None,
    n_mazes: int = 4,
) -> GenerationSnapshot:
    """Build a :class:`GenerationSnapshot` from the current state."""
    fitnesses = np.array([g.fitness for g in population])
    energies = np.array([g.energy for g in population])
    complexities = np.array([g.complexity for g in population])

    best_idx = int(np.argmax(fitnesses))
    best_genome = population[best_idx]

    # Success rate: genome solved all mazes.
    n_success = 0
    for gid, results in maze_results.items():
        if all(r.reached_goal for r in results):
            n_success += 1
    success_rate = n_success / max(len(population), 1)

    # Diversity: fraction of unique fitness values.
    unique = len(set(round(f, 6) for f in fitnesses))
    diversity = unique / max(len(population), 1)

    # Best path from best genome's first maze result.
    best_path = None
    if best_genome.genome_id in maze_results:
        runs = maze_results[best_genome.genome_id]
        if runs:
            best_path = runs[0].path

    return GenerationSnapshot(
        generation=generation,
        best_fitness=float(fitnesses[best_idx]),
        mean_fitness=float(fitnesses.mean()),
        best_energy=float(energies[best_idx]),
        mean_energy=float(energies.mean()),
        success_rate=success_rate,
        num_species=num_species,
        population_size=len(population),
        mean_complexity=float(complexities.mean()),
        max_complexity=float(complexities.max()),
        diversity=diversity,
        temperature=temperature,
        best_genome_id=best_genome.genome_id,
        best_path=best_path,
    )


__all__ = [
    "GenerationSnapshot",
    "EvolutionResult",
    "compute_snapshot",
]