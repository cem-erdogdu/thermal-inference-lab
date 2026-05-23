"""Full NEAT evolution loop.

Orchestrates population initialisation, evaluation, speciation,
selection (normal or Boltzmann), reproduction (crossover + mutation),
and diagnostics collection across generations.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from thermal_inference_lab.neat.genome import Genome, InnovationTracker
from thermal_inference_lab.neat.network import FeedForwardNetwork
from thermal_inference_lab.neat.mutation import mutate
from thermal_inference_lab.neat.crossover import crossover
from thermal_inference_lab.neat.speciation import SpeciationManager
from thermal_inference_lab.neat.selection import (
    select_normal,
    select_boltzmann,
    TemperatureSchedule,
)
from thermal_inference_lab.neat.maze.environment import GridMaze, N_INPUTS, N_OUTPUTS
from thermal_inference_lab.neat.maze.evaluation import (
    evaluate_population,
    MazeRunResult,
    MAZE_SUITE,
)
from thermal_inference_lab.neat.results import (
    EvolutionResult,
    GenerationSnapshot,
    compute_snapshot,
)


# ------------------------------------------------------------ config

class NeatConfig:
    """All tuneable knobs for a NEAT evolution run."""

    def __init__(
        self,
        *,
        population_size: int = 100,
        n_generations: int = 50,
        n_inputs: int = N_INPUTS,
        n_outputs: int = N_OUTPUTS,
        # Selection.
        selection_mode: str = "normal",  # "normal" or "boltzmann"
        elitism: int = 2,
        # Boltzmann.
        t_start: float = 5.0,
        t_end: float = 0.1,
        schedule_type: str = "exponential",
        # Mutation rates.
        weight_mutation_rate: float = 0.8,
        add_connection_rate: float = 0.05,
        add_node_rate: float = 0.03,
        toggle_rate: float = 0.02,
        perturb_std: float = 0.5,
        weight_range: float = 2.0,
        # Speciation.
        compatibility_threshold: float = 3.0,
        c1: float = 1.0,
        c2: float = 1.0,
        c3: float = 0.4,
        max_stagnation: int = 15,
        # Maze.
        max_steps: int = 100,
        complexity_penalty: float = 0.5,
        # Reproduction.
        crossover_rate: float = 0.75,
        interspecies_rate: float = 0.01,
        survival_threshold: float = 0.2,
        # Misc.
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.population_size = population_size
        self.n_generations = n_generations
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.selection_mode = selection_mode
        self.elitism = elitism
        self.t_start = t_start
        self.t_end = t_end
        self.schedule_type = schedule_type
        self.weight_mutation_rate = weight_mutation_rate
        self.add_connection_rate = add_connection_rate
        self.add_node_rate = add_node_rate
        self.toggle_rate = toggle_rate
        self.perturb_std = perturb_std
        self.weight_range = weight_range
        self.compatibility_threshold = compatibility_threshold
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.max_stagnation = max_stagnation
        self.max_steps = max_steps
        self.complexity_penalty = complexity_penalty
        self.crossover_rate = crossover_rate
        self.interspecies_rate = interspecies_rate
        self.survival_threshold = survival_threshold
        self.seed = seed
        self.verbose = verbose


# ---------------------------------------------------- evolution engine

def evolve(
    config: NeatConfig,
    mazes: Optional[List[GridMaze]] = None,
) -> EvolutionResult:
    """Run a full NEAT evolution and return structured results."""
    rng = np.random.default_rng(config.seed)
    tracker = InnovationTracker()
    if mazes is None:
        mazes = MAZE_SUITE

    # Reset genome id counter for reproducibility.
    Genome._next_genome_id = 0

    # ---- initialise population ----
    population: List[Genome] = []
    for _ in range(config.population_size):
        g = Genome.create_minimal(
            config.n_inputs, config.n_outputs, tracker, rng, weight_range=config.weight_range,
        )
        population.append(g)

    speciation_mgr = SpeciationManager(
        threshold=config.compatibility_threshold,
        c1=config.c1, c2=config.c2, c3=config.c3,
        max_stagnation=config.max_stagnation,
    )

    temp_schedule: Optional[TemperatureSchedule] = None
    if config.selection_mode == "boltzmann":
        temp_schedule = TemperatureSchedule(
            t_start=config.t_start,
            t_end=config.t_end,
            total_generations=config.n_generations,
            schedule_type=config.schedule_type,
        )

    result = EvolutionResult(selection_mode=config.selection_mode)
    all_best_genome: Optional[Genome] = None
    all_best_fitness = -float("inf")

    # ---- generation loop ----
    for gen in range(config.n_generations):
        # Evaluate.
        maze_results = evaluate_population(
            population, mazes,
            max_steps=config.max_steps,
            complexity_penalty=config.complexity_penalty,
        )

        # Speciate.
        speciation_mgr.speciate(population, rng)
        speciation_mgr.remove_stagnant()

        # Track best.
        gen_best = max(population, key=lambda g: g.fitness)
        if gen_best.fitness > all_best_fitness:
            all_best_fitness = gen_best.fitness
            all_best_genome = gen_best.copy()
            all_best_genome.fitness = gen_best.fitness
            all_best_genome.energy = gen_best.energy

        # Temperature.
        temperature: Optional[float] = None
        if temp_schedule is not None:
            temperature = temp_schedule.get_temperature(gen)
            result.temperature_history.append(temperature)

        # Snapshot.
        snap = compute_snapshot(
            gen, population, speciation_mgr.num_species,
            maze_results, temperature, n_mazes=len(mazes),
        )
        result.history.append(snap)
        if config.verbose:
            print(snap.summary_line())

        # ---- reproduction ----
        new_population: List[Genome] = []

        # Elites — preserve top genomes unchanged.
        sorted_pop = sorted(population, key=lambda g: g.fitness, reverse=True)
        for i in range(min(config.elitism, len(sorted_pop))):
            elite = sorted_pop[i].copy()
            elite.fitness = sorted_pop[i].fitness
            elite.energy = sorted_pop[i].energy
            new_population.append(elite)

        # Per-species reproduction.
        adjusted = speciation_mgr.adjusted_fitness()
        total_adj = sum(adjusted.values())
        species_list = list(speciation_mgr.species.values())

        # Compute offspring allocation per species.
        offspring_counts: Dict[int, int] = {}
        remaining = config.population_size - len(new_population)
        if total_adj > 0 and species_list:
            for sp in species_list:
                sp_adj = sum(adjusted.get(g.genome_id, 0.0) for g in sp.members)
                count = max(1, int(round(sp_adj / total_adj * remaining)))
                offspring_counts[sp.id] = count
            # Adjust to fill exactly.
            total_alloc = sum(offspring_counts.values())
            diff = remaining - total_alloc
            if diff != 0 and offspring_counts:
                biggest_sp = max(offspring_counts, key=lambda k: offspring_counts[k])
                offspring_counts[biggest_sp] = max(1, offspring_counts[biggest_sp] + diff)
        else:
            # Fallback: uniform allocation.
            if species_list:
                per = max(1, remaining // len(species_list))
                for sp in species_list:
                    offspring_counts[sp.id] = per

        for sp in species_list:
            n_offspring = offspring_counts.get(sp.id, 0)
            if n_offspring <= 0:
                continue
            members = sorted(sp.members, key=lambda g: g.fitness, reverse=True)
            # Keep top fraction as parents.
            n_parents = max(2, int(len(members) * config.survival_threshold))
            parents = members[:n_parents]

            for _ in range(n_offspring):
                if len(new_population) >= config.population_size:
                    break
                if rng.random() < config.crossover_rate and len(parents) >= 2:
                    # Crossover.
                    if rng.random() < config.interspecies_rate and len(population) > 2:
                        p1 = parents[int(rng.integers(len(parents)))]
                        p2 = population[int(rng.integers(len(population)))]
                    else:
                        idxs = rng.choice(len(parents), size=2, replace=False)
                        p1 = parents[idxs[0]]
                        p2 = parents[idxs[1]]
                    if p1.fitness >= p2.fitness:
                        child = crossover(p1, p2, rng)
                    else:
                        child = crossover(p2, p1, rng)
                else:
                    # Asexual — copy and mutate.
                    child = parents[int(rng.integers(len(parents)))].copy()

                mutate(
                    child, tracker, rng,
                    weight_mutation_rate=config.weight_mutation_rate,
                    add_connection_rate=config.add_connection_rate,
                    add_node_rate=config.add_node_rate,
                    toggle_rate=config.toggle_rate,
                    perturb_std=config.perturb_std,
                    weight_range=config.weight_range,
                )
                new_population.append(child)

        # Pad or trim.
        while len(new_population) < config.population_size:
            filler = Genome.create_minimal(
                config.n_inputs, config.n_outputs, tracker, rng,
                weight_range=config.weight_range,
            )
            new_population.append(filler)
        population = new_population[: config.population_size]

        # Reset per-generation innovation cache.
        tracker.reset_generation()

    # ---- final evaluation for result ----
    evaluate_population(
        population, mazes,
        max_steps=config.max_steps,
        complexity_penalty=config.complexity_penalty,
    )
    final_best = max(population, key=lambda g: g.fitness)
    if final_best.fitness > all_best_fitness:
        all_best_genome = final_best.copy()
        all_best_genome.fitness = final_best.fitness
        all_best_genome.energy = final_best.energy

    result.generations_run = config.n_generations
    result.best_genome = all_best_genome
    result.best_fitness = all_best_genome.fitness if all_best_genome else 0.0
    result.best_energy = all_best_genome.energy if all_best_genome else 0.0

    # Run best genome one more time to record maze results.
    if all_best_genome is not None:
        from thermal_inference_lab.neat.maze.evaluation import evaluate_genome
        _, _, best_runs = evaluate_genome(
            all_best_genome, mazes,
            max_steps=config.max_steps,
            complexity_penalty=config.complexity_penalty,
        )
        result.best_maze_results = {r.maze_name: r for r in best_runs}

    return result


__all__ = ["NeatConfig", "evolve"]