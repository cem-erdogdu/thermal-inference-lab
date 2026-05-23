"""Comprehensive tests for the NEAT neuroevolution system.

Covers: maze movement, wall collisions, goal detection, deterministic
rollouts, innovation tracking, mutation, crossover, disabled genes,
compatibility distance, speciation, Boltzmann selection probabilities,
temperature schedule behaviour, fixed-seed reproducibility, and
improvement on simple mazes.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

# ---- Genome / Innovation ----
from thermal_inference_lab.neat.genome import (
    ConnectionGene,
    Genome,
    InnovationTracker,
    NodeGene,
)
from thermal_inference_lab.neat.network import FeedForwardNetwork
from thermal_inference_lab.neat.mutation import (
    add_connection,
    add_node,
    mutate,
    mutate_weights,
    toggle_connection,
)
from thermal_inference_lab.neat.crossover import crossover
from thermal_inference_lab.neat.speciation import (
    SpeciationManager,
    compatibility_distance,
)
from thermal_inference_lab.neat.selection import (
    TemperatureSchedule,
    boltzmann_probabilities,
    exponential_schedule,
    linear_schedule,
    select_boltzmann,
    select_normal,
)
from thermal_inference_lab.neat.maze.environment import (
    BUILT_IN_MAZES,
    GridMaze,
    MazeAgent,
    N_INPUTS,
    N_OUTPUTS,
)
from thermal_inference_lab.neat.maze.evaluation import (
    evaluate_genome,
    evaluate_population,
    run_maze,
    MAZE_SUITE,
)
from thermal_inference_lab.neat.evolution import NeatConfig, evolve
from thermal_inference_lab.neat.results import EvolutionResult, compute_snapshot


# =====================================================================
# Helpers
# =====================================================================

def _make_rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_minimal_genome(seed: int = 0) -> tuple[Genome, InnovationTracker]:
    rng = _make_rng(seed)
    Genome._next_genome_id = 0
    tracker = InnovationTracker()
    g = Genome.create_minimal(N_INPUTS, N_OUTPUTS, tracker, rng)
    return g, tracker


def _simple_maze() -> GridMaze:
    return BUILT_IN_MAZES["simple"]()


# =====================================================================
# Maze environment tests
# =====================================================================

class TestMazeMovement:
    """Test basic agent movement in the maze."""

    def test_agent_starts_at_start(self):
        maze = _simple_maze()
        agent = MazeAgent(maze)
        assert agent.position == maze.start

    def test_move_to_empty_cell(self):
        maze = _simple_maze()
        agent = MazeAgent(maze)
        # Start is (1,1). Move right (action=3) to (1,2) which is empty.
        result = agent.step(3)
        assert result.position == (1, 2)
        assert not result.hit_wall
        assert agent.position == (1, 2)

    def test_wall_collision_stays_in_place(self):
        maze = _simple_maze()
        agent = MazeAgent(maze)
        # Start is (1,1). Move up (action=0) to (0,1) which is a wall.
        result = agent.step(0)
        assert result.hit_wall
        assert agent.position == maze.start  # stays
        assert agent.wall_hits == 1

    def test_goal_detection(self):
        """Agent reaches the goal cell."""
        maze = _simple_maze()
        agent = MazeAgent(maze)
        # Navigate from (1,1) to goal (3,3) via a known path.
        # (1,1)->(2,1)->(3,1)->(3,2)->(3,3)
        actions = [1, 1, 3, 3]  # down, down, right, right
        for a in actions:
            result = agent.step(a)
        # (3,3) is the goal cell.
        assert result.reached_goal
        assert agent.reached_goal

    def test_path_recording(self):
        maze = _simple_maze()
        agent = MazeAgent(maze)
        agent.step(3)  # right
        agent.step(3)  # right
        assert len(agent.path) == 3  # start + 2 steps
        assert agent.path[0] == maze.start

    def test_done_after_max_steps(self):
        maze = _simple_maze()
        agent = MazeAgent(maze, max_steps=3)
        for _ in range(3):
            agent.step(0)  # hit wall repeatedly
        assert agent.done

    def test_revisit_count(self):
        maze = _simple_maze()
        agent = MazeAgent(maze)
        agent.step(3)  # right to (1,2)
        agent.step(2)  # left back to (1,1)
        assert agent.revisit_count >= 1


class TestMazeWallCollision:
    """Test wall collision edge cases."""

    def test_out_of_bounds_is_wall(self):
        maze = _simple_maze()
        assert maze.is_wall(-1, 0)
        assert maze.is_wall(0, -1)
        assert maze.is_wall(maze.rows, 0)
        assert maze.is_wall(0, maze.cols)

    def test_wall_cell_detected(self):
        maze = _simple_maze()
        # (0,0) is a wall in the simple maze.
        assert maze.is_wall(0, 0)

    def test_repeated_wall_hits(self):
        maze = _simple_maze()
        agent = MazeAgent(maze)
        for _ in range(5):
            agent.step(0)  # up — always hits wall from (1,1)
        assert agent.wall_hits == 5
        assert agent.position == maze.start


class TestMazeGoalDetection:
    """Goal detection and input sensor tests."""

    def test_goal_position_correct(self):
        maze = _simple_maze()
        assert maze.goal == (3, 3)

    def test_inputs_have_correct_shape(self):
        maze = _simple_maze()
        agent = MazeAgent(maze)
        inputs = agent.get_inputs()
        assert inputs.shape == (N_INPUTS,)

    def test_wall_sensors_at_corner(self):
        maze = _simple_maze()
        agent = MazeAgent(maze)
        inputs = agent.get_inputs()
        # At (1,1): up=(0,1)=wall, left=(1,0)=wall
        assert inputs[0] == 1.0  # wall up
        assert inputs[2] == 1.0  # wall left

    def test_goal_direction_positive(self):
        maze = _simple_maze()
        agent = MazeAgent(maze)
        inputs = agent.get_inputs()
        # Goal is at (3,3), agent at (1,1). dx>0, dy>0.
        assert inputs[4] > 0  # dx (col direction)
        assert inputs[5] > 0  # dy (row direction)


class TestDeterministicRollout:
    """Test that maze rollouts are deterministic given a fixed network."""

    def test_same_network_same_result(self):
        genome, _ = _make_minimal_genome(seed=99)
        maze = _simple_maze()
        net = FeedForwardNetwork.from_genome(genome)
        r1 = run_maze(net, maze, max_steps=50)
        r2 = run_maze(net, maze, max_steps=50)
        assert r1.path == r2.path
        assert r1.steps_taken == r2.steps_taken
        assert r1.reached_goal == r2.reached_goal


# =====================================================================
# Innovation tracking tests
# =====================================================================

class TestInnovationTracker:

    def test_same_connection_same_innovation(self):
        tracker = InnovationTracker()
        i1 = tracker.get_innovation(0, 5)
        i2 = tracker.get_innovation(0, 5)
        assert i1 == i2

    def test_different_connections_different_innovations(self):
        tracker = InnovationTracker()
        i1 = tracker.get_innovation(0, 5)
        i2 = tracker.get_innovation(1, 5)
        assert i1 != i2

    def test_counter_increments(self):
        tracker = InnovationTracker()
        tracker.get_innovation(0, 5)
        tracker.get_innovation(1, 5)
        assert tracker.counter == 2

    def test_reset_clears_history(self):
        tracker = InnovationTracker()
        tracker.get_innovation(0, 5)
        tracker.reset_generation()
        assert tracker.history == {}
        # Same pair now gets a new innovation number.
        i2 = tracker.get_innovation(0, 5)
        assert i2 == 1  # counter wasn't reset, only history


# =====================================================================
# Mutation tests
# =====================================================================

class TestMutation:

    def test_mutate_weights_changes_something(self):
        genome, tracker = _make_minimal_genome()
        rng = _make_rng(1)
        old_weights = {k: c.weight for k, c in genome.connections.items()}
        mutate_weights(genome, rng)
        new_weights = {k: c.weight for k, c in genome.connections.items()}
        assert old_weights != new_weights

    def test_add_connection_increases_count(self):
        genome, tracker = _make_minimal_genome()
        rng = _make_rng(2)
        n_before = len(genome.connections)
        # Add a hidden node first so there's room for a new connection.
        add_node(genome, tracker, rng)
        added = add_connection(genome, tracker, rng)
        if added:
            assert len(genome.connections) > n_before + 2  # +2 from add_node

    def test_add_node_splits_connection(self):
        genome, tracker = _make_minimal_genome()
        rng = _make_rng(3)
        n_hidden_before = len(genome.hidden_nodes)
        n_conn_before = len(genome.connections)
        result = add_node(genome, tracker, rng)
        assert result is True
        assert len(genome.hidden_nodes) == n_hidden_before + 1
        # One connection disabled, two new ones added.
        assert len(genome.connections) == n_conn_before + 2
        # Exactly one disabled connection.
        disabled = [c for c in genome.connections.values() if not c.enabled]
        assert len(disabled) >= 1

    def test_toggle_connection(self):
        genome, tracker = _make_minimal_genome()
        rng = _make_rng(4)
        states_before = {k: c.enabled for k, c in genome.connections.items()}
        toggle_connection(genome, rng)
        states_after = {k: c.enabled for k, c in genome.connections.items()}
        assert states_before != states_after

    def test_composite_mutate_runs(self):
        """The composite mutate function doesn't crash."""
        genome, tracker = _make_minimal_genome()
        rng = _make_rng(5)
        mutate(genome, tracker, rng)
        assert len(genome.nodes) > 0


# =====================================================================
# Crossover tests
# =====================================================================

class TestCrossover:

    def test_child_has_all_input_output_nodes(self):
        g1, tracker = _make_minimal_genome(seed=10)
        g2 = Genome.create_minimal(N_INPUTS, N_OUTPUTS, tracker, _make_rng(11))
        rng = _make_rng(12)
        child = crossover(g1, g2, rng)
        assert set(child.input_nodes) == set(g1.input_nodes)
        assert set(child.output_nodes) == set(g1.output_nodes)

    def test_matching_genes_from_either_parent(self):
        g1, tracker = _make_minimal_genome(seed=20)
        g2 = Genome.create_minimal(N_INPUTS, N_OUTPUTS, tracker, _make_rng(21))
        g1.fitness = 10.0
        g2.fitness = 10.0  # equal fitness
        rng = _make_rng(22)
        child = crossover(g1, g2, rng)
        # Child should have connections from matching innovations.
        matching = set(g1.connections) & set(g2.connections)
        for innov in matching:
            assert innov in child.connections

    def test_fitter_parent_disjoint_excess(self):
        g1, tracker = _make_minimal_genome(seed=30)
        g2 = Genome.create_minimal(N_INPUTS, N_OUTPUTS, tracker, _make_rng(31))
        rng = _make_rng(32)
        # Add extra structure to g1.
        add_node(g1, tracker, rng)
        add_connection(g1, tracker, rng)
        g1.fitness = 20.0
        g2.fitness = 5.0
        child = crossover(g1, g2, rng)
        # Child should have the extra connections from fitter parent g1.
        for innov in g1.connections:
            assert innov in child.connections

    def test_disabled_gene_inheritance(self):
        g1, tracker = _make_minimal_genome(seed=40)
        g2 = Genome.create_minimal(N_INPUTS, N_OUTPUTS, tracker, _make_rng(41))
        # Disable a gene in g1.
        first_innov = next(iter(g1.connections))
        g1.connections[first_innov].enabled = False
        g1.fitness = 10.0
        g2.fitness = 10.0
        # Run crossover many times; the disabled gene should sometimes be disabled.
        disabled_count = 0
        for seed in range(50, 150):
            child = crossover(g1, g2, _make_rng(seed))
            if first_innov in child.connections and not child.connections[first_innov].enabled:
                disabled_count += 1
        # With disabled_gene_inherit_rate=0.75, expect a good fraction disabled.
        assert disabled_count > 10


# =====================================================================
# Speciation tests
# =====================================================================

class TestCompatibilityDistance:

    def test_identical_genomes_zero_distance(self):
        g1, tracker = _make_minimal_genome(seed=60)
        g2 = g1.copy()
        # Same structure and weights.
        for innov in g2.connections:
            g2.connections[innov].weight = g1.connections[innov].weight
        d = compatibility_distance(g1, g2)
        assert d == pytest.approx(0.0)

    def test_different_weights_nonzero(self):
        g1, tracker = _make_minimal_genome(seed=61)
        g2 = g1.copy()
        for c in g2.connections.values():
            c.weight += 5.0
        d = compatibility_distance(g1, g2)
        assert d > 0.0

    def test_extra_genes_increase_distance(self):
        g1, tracker = _make_minimal_genome(seed=62)
        g2 = g1.copy()
        rng = _make_rng(63)
        add_node(g2, tracker, rng)
        d_before = compatibility_distance(g1, g1.copy())
        d_after = compatibility_distance(g1, g2)
        assert d_after > d_before


class TestSpeciation:

    def test_initial_speciation(self):
        Genome._next_genome_id = 0
        tracker = InnovationTracker()
        rng = _make_rng(70)
        pop = [Genome.create_minimal(N_INPUTS, N_OUTPUTS, tracker, rng)
               for _ in range(10)]
        mgr = SpeciationManager(threshold=3.0)
        mgr.speciate(pop, rng)
        assert mgr.num_species >= 1
        # Every genome has a species.
        for g in pop:
            assert g.species_id >= 0

    def test_divergent_genomes_form_multiple_species(self):
        Genome._next_genome_id = 0
        tracker = InnovationTracker()
        rng = _make_rng(71)
        pop = []
        for _ in range(10):
            g = Genome.create_minimal(N_INPUTS, N_OUTPUTS, tracker, rng)
            # Make them divergent.
            for _ in range(5):
                add_node(g, tracker, rng)
                add_connection(g, tracker, rng)
                mutate_weights(g, rng, perturb_std=3.0)
            pop.append(g)
        mgr = SpeciationManager(threshold=1.0)
        mgr.speciate(pop, rng)
        assert mgr.num_species > 1


# =====================================================================
# Selection tests
# =====================================================================

class TestBoltzmannSelection:

    def test_probabilities_sum_to_one(self):
        energies = np.array([-10.0, -5.0, 0.0, 5.0])
        probs = boltzmann_probabilities(energies, temperature=1.0)
        assert probs.sum() == pytest.approx(1.0)

    def test_lower_energy_higher_probability(self):
        energies = np.array([0.0, 10.0])
        probs = boltzmann_probabilities(energies, temperature=1.0)
        assert probs[0] > probs[1]

    def test_zero_temperature_picks_best(self):
        energies = np.array([5.0, 1.0, 10.0])
        probs = boltzmann_probabilities(energies, temperature=0.0)
        assert probs[1] == 1.0  # index 1 has lowest energy

    def test_high_temperature_nearly_uniform(self):
        energies = np.array([0.0, 1.0, 2.0])
        probs = boltzmann_probabilities(energies, temperature=1e6)
        assert all(abs(p - 1.0 / 3) < 0.01 for p in probs)

    def test_select_boltzmann_returns_correct_count(self):
        Genome._next_genome_id = 0
        pop = []
        for i in range(10):
            g = Genome()
            g.energy = float(i)
            g.fitness = 10.0 - i
            pop.append(g)
        rng = _make_rng(80)
        selected = select_boltzmann(pop, 5, rng, temperature=1.0)
        assert len(selected) == 5


class TestTemperatureSchedule:

    def test_linear_schedule_endpoints(self):
        t0 = linear_schedule(10.0, 1.0, 0, 10)
        t9 = linear_schedule(10.0, 1.0, 9, 10)
        assert t0 == pytest.approx(10.0)
        assert t9 == pytest.approx(1.0)

    def test_exponential_schedule_endpoints(self):
        t0 = exponential_schedule(10.0, 0.1, 0, 10)
        t9 = exponential_schedule(10.0, 0.1, 9, 10)
        assert t0 == pytest.approx(10.0)
        assert t9 == pytest.approx(0.1)

    def test_exponential_monotone_decreasing(self):
        temps = [exponential_schedule(10.0, 0.1, i, 20) for i in range(20)]
        assert all(temps[i] >= temps[i + 1] for i in range(len(temps) - 1))

    def test_temperature_schedule_tracker(self):
        ts = TemperatureSchedule(t_start=5.0, t_end=0.5, total_generations=10,
                                 schedule_type="linear")
        for i in range(10):
            ts.get_temperature(i)
        assert len(ts.history) == 10
        assert ts.history[0] == pytest.approx(5.0)
        assert ts.history[-1] == pytest.approx(0.5)


# =====================================================================
# Network tests
# =====================================================================

class TestNetwork:

    def test_minimal_network_output_shape(self):
        genome, _ = _make_minimal_genome()
        net = FeedForwardNetwork.from_genome(genome)
        out = net.activate(np.zeros(N_INPUTS))
        assert out.shape == (N_OUTPUTS,)

    def test_output_in_sigmoid_range(self):
        genome, _ = _make_minimal_genome()
        net = FeedForwardNetwork.from_genome(genome)
        out = net.activate(np.random.randn(N_INPUTS))
        assert all(0.0 <= v <= 1.0 for v in out)

    def test_hidden_node_network(self):
        genome, tracker = _make_minimal_genome(seed=90)
        rng = _make_rng(91)
        add_node(genome, tracker, rng)
        net = FeedForwardNetwork.from_genome(genome)
        out = net.activate(np.zeros(N_INPUTS))
        assert out.shape == (N_OUTPUTS,)


# =====================================================================
# Fixed-seed reproducibility
# =====================================================================

class TestReproducibility:

    def test_same_seed_same_population(self):
        """Two evolution runs with the same seed produce identical fitness curves."""
        mazes = [BUILT_IN_MAZES["simple"]()]
        cfg = NeatConfig(
            population_size=20, n_generations=5, seed=123,
            verbose=False, max_steps=30,
        )
        Genome._next_genome_id = 0
        r1 = evolve(cfg, mazes)
        Genome._next_genome_id = 0
        r2 = evolve(cfg, mazes)
        np.testing.assert_array_almost_equal(
            r1.fitness_curve, r2.fitness_curve, decimal=10,
        )

    def test_different_seeds_differ(self):
        mazes = [BUILT_IN_MAZES["simple"]()]
        Genome._next_genome_id = 0
        r1 = evolve(NeatConfig(population_size=20, n_generations=5, seed=1,
                                verbose=False, max_steps=30), mazes)
        Genome._next_genome_id = 0
        r2 = evolve(NeatConfig(population_size=20, n_generations=5, seed=999,
                                verbose=False, max_steps=30), mazes)
        # Very unlikely to be identical.
        assert not np.allclose(r1.fitness_curve, r2.fitness_curve)


# =====================================================================
# Evolution improvement test
# =====================================================================

class TestEvolutionImprovement:

    def test_fitness_improves_on_simple_maze(self):
        """Evolution should improve fitness over 15 generations on the simple maze."""
        mazes = [BUILT_IN_MAZES["simple"]()]
        Genome._next_genome_id = 0
        cfg = NeatConfig(
            population_size=40,
            n_generations=15,
            seed=42,
            verbose=False,
            max_steps=50,
        )
        result = evolve(cfg, mazes)
        # Best fitness at the end should be greater than at the start.
        assert result.fitness_curve[-1] >= result.fitness_curve[0]

    def test_result_object_complete(self):
        mazes = [BUILT_IN_MAZES["simple"]()]
        Genome._next_genome_id = 0
        cfg = NeatConfig(
            population_size=20, n_generations=5, seed=42,
            verbose=False, max_steps=30,
        )
        result = evolve(cfg, mazes)
        assert result.generations_run == 5
        assert result.best_genome is not None
        assert len(result.history) == 5
        assert result.best_fitness >= 0.0
        assert result.best_maze_results is not None
        assert "simple" in result.best_maze_results

    def test_boltzmann_mode_runs(self):
        """Boltzmann selection mode completes without error."""
        mazes = [BUILT_IN_MAZES["simple"]()]
        Genome._next_genome_id = 0
        cfg = NeatConfig(
            population_size=20, n_generations=5, seed=42,
            selection_mode="boltzmann", verbose=False, max_steps=30,
        )
        result = evolve(cfg, mazes)
        assert result.selection_mode == "boltzmann"
        assert len(result.temperature_history) == 5
        assert result.best_fitness >= 0.0


# =====================================================================
# Normal selection test
# =====================================================================

class TestNormalSelection:

    def test_elites_always_selected(self):
        Genome._next_genome_id = 0
        pop = []
        for i in range(10):
            g = Genome()
            g.fitness = float(i)
            pop.append(g)
        rng = _make_rng(100)
        selected = select_normal(pop, 5, rng, elitism=2)
        # Top 2 genomes (fitness 9 and 8) must be in the selection.
        top_ids = {pop[-1].genome_id, pop[-2].genome_id}
        sel_ids = {g.genome_id for g in selected[:2]}
        assert top_ids == sel_ids