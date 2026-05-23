"""NEAT genome representation.

A genome consists of a set of node genes and connection genes.  Each
connection carries a globally unique *innovation number* assigned by
:class:`InnovationTracker`, which allows crossover to align homologous
structures across different topologies.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------ genes

@dataclass
class NodeGene:
    """A single node in the NEAT network."""
    id: int
    type: str  # 'input', 'output', 'hidden', 'bias'
    activation: str = "sigmoid"  # 'sigmoid', 'tanh', 'relu', 'identity'


@dataclass
class ConnectionGene:
    """A directed connection between two nodes."""
    innovation: int
    in_node: int
    out_node: int
    weight: float = 0.0
    enabled: bool = True


# --------------------------------------------------------- innovation tracker

class InnovationTracker:
    """Global registry that assigns unique innovation numbers.

    When a structural mutation creates a new connection between nodes
    ``(in_id, out_id)``, the tracker returns the same innovation number
    for any genome that independently discovers the same connection
    within the same generation.
    """

    def __init__(self, start: int = 0) -> None:
        self._counter: int = start
        self._history: Dict[Tuple[int, int], int] = {}

    @property
    def counter(self) -> int:
        return self._counter

    @property
    def history(self) -> Dict[Tuple[int, int], int]:
        return dict(self._history)

    def get_innovation(self, in_node: int, out_node: int) -> int:
        """Return the innovation number for a connection, creating one if new."""
        key = (in_node, out_node)
        if key not in self._history:
            self._history[key] = self._counter
            self._counter += 1
        return self._history[key]

    def reset_generation(self) -> None:
        """Clear per-generation cache so new generations get fresh checks."""
        self._history.clear()


# ------------------------------------------------------------------ genome

class Genome:
    """A NEAT genome: a variable-topology neural network blueprint.

    Attributes
    ----------
    genome_id : int
        Unique identifier for this genome.
    nodes : dict[int, NodeGene]
        Node genes keyed by node id.
    connections : dict[int, ConnectionGene]
        Connection genes keyed by innovation number.
    fitness : float
        Fitness assigned by the evaluation function.
    energy : float
        Energy value for Boltzmann selection (lower is better).
    species_id : int
        Species this genome belongs to (-1 = unassigned).
    """

    _next_genome_id: int = 0

    def __init__(self, genome_id: Optional[int] = None) -> None:
        if genome_id is None:
            genome_id = Genome._next_genome_id
            Genome._next_genome_id += 1
        else:
            Genome._next_genome_id = max(Genome._next_genome_id, genome_id + 1)
        self.genome_id: int = genome_id
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[int, ConnectionGene] = {}
        self.fitness: float = 0.0
        self.energy: float = 0.0
        self.species_id: int = -1

    # -------------------------------------------------------- factories

    @classmethod
    def create_minimal(
        cls,
        n_inputs: int,
        n_outputs: int,
        tracker: InnovationTracker,
        rng: np.random.Generator,
        *,
        include_bias: bool = True,
        weight_range: float = 1.0,
    ) -> "Genome":
        """Create a minimal fully-connected genome (inputs -> outputs).

        Node ids: inputs 0..n_inputs-1, bias n_inputs (if included),
        outputs start at n_inputs + (1 if bias else 0).
        """
        g = cls()
        nid = 0
        input_ids: List[int] = []
        for _ in range(n_inputs):
            g.nodes[nid] = NodeGene(id=nid, type="input", activation="identity")
            input_ids.append(nid)
            nid += 1
        bias_id: Optional[int] = None
        if include_bias:
            g.nodes[nid] = NodeGene(id=nid, type="bias", activation="identity")
            bias_id = nid
            nid += 1
        output_ids: List[int] = []
        for _ in range(n_outputs):
            g.nodes[nid] = NodeGene(id=nid, type="output", activation="sigmoid")
            output_ids.append(nid)
            nid += 1
        # Connect every input (and bias) to every output.
        source_ids = input_ids + ([bias_id] if bias_id is not None else [])
        for src in source_ids:
            for dst in output_ids:
                innov = tracker.get_innovation(src, dst)
                w = rng.uniform(-weight_range, weight_range)
                g.connections[innov] = ConnectionGene(
                    innovation=innov,
                    in_node=src,
                    out_node=dst,
                    weight=float(w),
                    enabled=True,
                )
        return g

    # ------------------------------------------------------- properties

    @property
    def input_nodes(self) -> List[int]:
        return sorted(n.id for n in self.nodes.values() if n.type == "input")

    @property
    def output_nodes(self) -> List[int]:
        return sorted(n.id for n in self.nodes.values() if n.type == "output")

    @property
    def hidden_nodes(self) -> List[int]:
        return sorted(n.id for n in self.nodes.values() if n.type == "hidden")

    @property
    def bias_nodes(self) -> List[int]:
        return sorted(n.id for n in self.nodes.values() if n.type == "bias")

    @property
    def max_node_id(self) -> int:
        return max(self.nodes.keys()) if self.nodes else -1

    @property
    def n_enabled_connections(self) -> int:
        return sum(1 for c in self.connections.values() if c.enabled)

    @property
    def complexity(self) -> float:
        """A scalar measure of genome complexity (nodes + enabled connections)."""
        return float(len(self.hidden_nodes) + self.n_enabled_connections)

    # -------------------------------------------------------------- copy

    def copy(self) -> "Genome":
        """Return a deep copy with a new genome id."""
        g = Genome()
        g.nodes = {k: copy.copy(v) for k, v in self.nodes.items()}
        g.connections = {k: copy.copy(v) for k, v in self.connections.items()}
        g.fitness = self.fitness
        g.energy = self.energy
        g.species_id = self.species_id
        return g

    def __repr__(self) -> str:
        return (
            f"Genome(id={self.genome_id}, nodes={len(self.nodes)}, "
            f"conns={len(self.connections)}, enabled={self.n_enabled_connections}, "
            f"fitness={self.fitness:.4f}, energy={self.energy:.4f})"
        )


__all__ = [
    "NodeGene",
    "ConnectionGene",
    "InnovationTracker",
    "Genome",
]