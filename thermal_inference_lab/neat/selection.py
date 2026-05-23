"""Selection operators for NEAT evolution.

Two modes are supported:

1. **Normal (fitness-proportionate / truncation)** — the standard NEAT
   approach where offspring counts are proportional to shared fitness.
2. **Boltzmann (energy-based)** — each genome has an *energy* (lower =
   better); the selection probability follows the Boltzmann distribution

   .. math::

       p_i = \\frac{e^{-E_i / T}}{\\sum_j e^{-E_j / T}}

   with a temperature *T* that is reduced over generations according to
   an annealing schedule.  At high *T* selection is nearly uniform
   (exploration); as *T* → 0 it collapses onto the lowest-energy
   genome (exploitation).

Temperature schedules
---------------------
- ``linear``  — T(g) = T0 - (T0 - T_end) * g / G
- ``exponential`` — T(g) = T0 * (T_end / T0)^(g / G)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from thermal_inference_lab.neat.genome import Genome


# ----------------------------------------------------------- schedules

def linear_schedule(t_start: float, t_end: float, step: int, total_steps: int) -> float:
    """Linearly interpolate temperature from *t_start* to *t_end*."""
    if total_steps <= 1:
        return t_end
    frac = step / (total_steps - 1)
    return t_start + frac * (t_end - t_start)


def exponential_schedule(t_start: float, t_end: float, step: int, total_steps: int) -> float:
    """Exponentially decay temperature from *t_start* to *t_end*."""
    if total_steps <= 1:
        return t_end
    if t_start <= 0 or t_end <= 0:
        raise ValueError("Temperatures must be positive for exponential schedule")
    frac = step / (total_steps - 1)
    return t_start * (t_end / t_start) ** frac


SCHEDULES = {
    "linear": linear_schedule,
    "exponential": exponential_schedule,
}


# -------------------------------------------------------- Boltzmann probs

def boltzmann_probabilities(energies: np.ndarray, temperature: float) -> np.ndarray:
    """Compute Boltzmann selection probabilities from energy values.

    Lower energy → higher probability.  Returns a probability vector
    that sums to 1.
    """
    if temperature <= 0:
        # Zero temperature: all mass on minimum energy.
        probs = np.zeros_like(energies, dtype=np.float64)
        probs[np.argmin(energies)] = 1.0
        return probs
    log_weights = -energies / temperature
    # Numerically stable softmax.
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    total = weights.sum()
    if total == 0.0:
        return np.ones_like(weights) / len(weights)
    return weights / total


# ---------------------------------------------------- selection modes

def select_normal(
    population: List[Genome],
    n: int,
    rng: np.random.Generator,
    *,
    elitism: int = 1,
) -> List[Genome]:
    """Fitness-proportionate selection with optional elitism.

    Returns *n* parent genomes.  The top *elitism* genomes are
    guaranteed to be included first.
    """
    if not population:
        return []
    sorted_pop = sorted(population, key=lambda g: g.fitness, reverse=True)
    selected: List[Genome] = []

    # Elites.
    for i in range(min(elitism, len(sorted_pop), n)):
        selected.append(sorted_pop[i])

    remaining = n - len(selected)
    if remaining <= 0:
        return selected[:n]

    # Fitness-proportionate (shift so minimum is >= 0).
    fitnesses = np.array([g.fitness for g in population], dtype=np.float64)
    shifted = fitnesses - fitnesses.min()
    total = shifted.sum()
    if total == 0:
        probs = np.ones(len(population)) / len(population)
    else:
        probs = shifted / total

    idxs = rng.choice(len(population), size=remaining, replace=True, p=probs)
    for idx in idxs:
        selected.append(population[idx])
    return selected


def select_boltzmann(
    population: List[Genome],
    n: int,
    rng: np.random.Generator,
    *,
    temperature: float = 1.0,
    elitism: int = 1,
) -> List[Genome]:
    """Energy-based Boltzmann selection.

    Lower energy → higher selection probability at the current
    *temperature*.  The top *elitism* genomes (by lowest energy) are
    guaranteed to be included.
    """
    if not population:
        return []
    # Sort by energy ascending (best first).
    sorted_pop = sorted(population, key=lambda g: g.energy)
    selected: List[Genome] = []

    for i in range(min(elitism, len(sorted_pop), n)):
        selected.append(sorted_pop[i])

    remaining = n - len(selected)
    if remaining <= 0:
        return selected[:n]

    energies = np.array([g.energy for g in population], dtype=np.float64)
    probs = boltzmann_probabilities(energies, temperature)
    idxs = rng.choice(len(population), size=remaining, replace=True, p=probs)
    for idx in idxs:
        selected.append(population[idx])
    return selected


# ----------------------------------------------------- temperature tracker

@dataclass
class TemperatureSchedule:
    """Track temperature across generations for Boltzmann selection."""
    t_start: float = 5.0
    t_end: float = 0.1
    total_generations: int = 100
    schedule_type: str = "exponential"
    history: List[float] = field(default_factory=list)

    def get_temperature(self, generation: int) -> float:
        schedule_fn = SCHEDULES.get(self.schedule_type, exponential_schedule)
        t = schedule_fn(self.t_start, self.t_end, generation, self.total_generations)
        self.history.append(t)
        return t

    def reset(self) -> None:
        self.history.clear()


__all__ = [
    "boltzmann_probabilities",
    "linear_schedule",
    "exponential_schedule",
    "select_normal",
    "select_boltzmann",
    "TemperatureSchedule",
    "SCHEDULES",
]