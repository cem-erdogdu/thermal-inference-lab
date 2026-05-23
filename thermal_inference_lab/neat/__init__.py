"""NEAT neuroevolution with energy-based Boltzmann selection.

A from-scratch implementation of NeuroEvolution of Augmenting
Topologies (NEAT) using only NumPy and the Python standard library.
Includes a grid-maze environment for evaluating evolved agents and
a Boltzmann energy-based selection mode with temperature annealing.
"""

from thermal_inference_lab.neat.genome import (
    NodeGene,
    ConnectionGene,
    InnovationTracker,
    Genome,
)
from thermal_inference_lab.neat.network import FeedForwardNetwork
from thermal_inference_lab.neat.mutation import mutate
from thermal_inference_lab.neat.crossover import crossover
from thermal_inference_lab.neat.speciation import (
    compatibility_distance,
    Species,
    SpeciationManager,
)
from thermal_inference_lab.neat.selection import (
    boltzmann_probabilities,
    select_normal,
    select_boltzmann,
    TemperatureSchedule,
)
from thermal_inference_lab.neat.results import (
    GenerationSnapshot,
    EvolutionResult,
)
from thermal_inference_lab.neat.evolution import NeatConfig, evolve

__all__ = [
    "NodeGene",
    "ConnectionGene",
    "InnovationTracker",
    "Genome",
    "FeedForwardNetwork",
    "mutate",
    "crossover",
    "compatibility_distance",
    "Species",
    "SpeciationManager",
    "boltzmann_probabilities",
    "select_normal",
    "select_boltzmann",
    "TemperatureSchedule",
    "GenerationSnapshot",
    "EvolutionResult",
    "NeatConfig",
    "evolve",
]