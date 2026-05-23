"""Multi-maze genome evaluation for NEAT.

Each genome is tested on every maze in the suite.  The composite
fitness rewards:

- **Reaching the goal** — large bonus.
- **Speed** — completing faster yields a higher score.
- **Wall avoidance** — each wall hit is penalised.
- **Loop avoidance** — revisiting cells is penalised.
- **Progress** — closing distance to the goal is rewarded.
- **Complexity penalty** — discourages gratuitous network growth.

Energy
------
The energy is derived as ``E = -fitness`` (so lower energy = better
performance), matching the Boltzmann selection convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from thermal_inference_lab.neat.genome import Genome
from thermal_inference_lab.neat.network import FeedForwardNetwork
from thermal_inference_lab.neat.maze.environment import (
    GridMaze,
    MazeAgent,
    BUILT_IN_MAZES,
    N_INPUTS,
    N_OUTPUTS,
)


# -------------------------------------------------------- default suite

MAZE_SUITE: List[GridMaze] = [fn() for fn in BUILT_IN_MAZES.values()]


# ----------------------------------------------------- evaluation result

@dataclass
class MazeRunResult:
    maze_name: str
    reached_goal: bool
    steps_taken: int
    wall_hits: int
    unique_cells: int
    revisits: int
    final_distance: float
    initial_distance: float
    path: List[Tuple[int, int]] = field(default_factory=list)
    score: float = 0.0


# ---------------------------------------------------------- scoring

def score_run(
    result: MazeRunResult,
    max_steps: int,
    *,
    goal_bonus: float = 100.0,
    speed_weight: float = 50.0,
    wall_penalty: float = 2.0,
    revisit_penalty: float = 1.0,
    progress_weight: float = 20.0,
) -> float:
    """Compute the scalar score for a single maze run."""
    score = 0.0
    # Goal bonus.
    if result.reached_goal:
        score += goal_bonus
        # Speed bonus (fraction of time saved).
        speed_frac = max(0.0, 1.0 - result.steps_taken / max_steps)
        score += speed_weight * speed_frac
    # Progress (even if goal not reached).
    if result.initial_distance > 0:
        progress = (result.initial_distance - result.final_distance) / result.initial_distance
        score += progress_weight * max(0.0, progress)
    # Penalties.
    score -= wall_penalty * result.wall_hits
    score -= revisit_penalty * result.revisits
    return max(score, 0.0)


# --------------------------------------------------------- run one maze

def run_maze(
    network: FeedForwardNetwork,
    maze: GridMaze,
    max_steps: int = 100,
) -> MazeRunResult:
    """Run *network* in *maze* for up to *max_steps*."""
    agent = MazeAgent(maze, max_steps=max_steps)
    initial_dist = float(maze.manhattan_distance(agent.position))

    while not agent.done:
        inputs = agent.get_inputs()
        outputs = network.activate(inputs)
        action = int(np.argmax(outputs))
        agent.step(action)

    final_dist = float(maze.manhattan_distance(agent.position))
    result = MazeRunResult(
        maze_name=maze.name,
        reached_goal=agent.reached_goal,
        steps_taken=agent.step_count,
        wall_hits=agent.wall_hits,
        unique_cells=agent.unique_cells_visited,
        revisits=agent.revisit_count,
        final_distance=final_dist,
        initial_distance=initial_dist,
        path=list(agent.path),
    )
    result.score = score_run(result, max_steps)
    return result


# ----------------------------------------------------- evaluate genome

def evaluate_genome(
    genome: Genome,
    mazes: Optional[List[GridMaze]] = None,
    *,
    max_steps: int = 100,
    complexity_penalty: float = 0.5,
) -> Tuple[float, float, List[MazeRunResult]]:
    """Evaluate *genome* on all mazes.

    Returns ``(fitness, energy, run_results)``.
    """
    if mazes is None:
        mazes = MAZE_SUITE
    network = FeedForwardNetwork.from_genome(genome)
    results: List[MazeRunResult] = []
    total_score = 0.0
    for maze in mazes:
        r = run_maze(network, maze, max_steps=max_steps)
        results.append(r)
        total_score += r.score

    # Average across mazes.
    avg_score = total_score / max(len(mazes), 1)
    # Complexity penalty.
    penalty = complexity_penalty * genome.complexity
    fitness = max(0.0, avg_score - penalty)
    energy = -fitness  # Lower energy = better.

    genome.fitness = fitness
    genome.energy = energy
    return fitness, energy, results


# -------------------------------------------------- evaluate population

def evaluate_population(
    population: List[Genome],
    mazes: Optional[List[GridMaze]] = None,
    *,
    max_steps: int = 100,
    complexity_penalty: float = 0.5,
) -> Dict[int, List[MazeRunResult]]:
    """Evaluate every genome in *population*.

    Returns a dict mapping genome_id → list of MazeRunResult.
    """
    all_results: Dict[int, List[MazeRunResult]] = {}
    for genome in population:
        _, _, results = evaluate_genome(
            genome, mazes, max_steps=max_steps,
            complexity_penalty=complexity_penalty,
        )
        all_results[genome.genome_id] = results
    return all_results


__all__ = [
    "MazeRunResult",
    "score_run",
    "run_maze",
    "evaluate_genome",
    "evaluate_population",
    "MAZE_SUITE",
]