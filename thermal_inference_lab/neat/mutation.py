"""NEAT mutation operators.

All mutations accept an ``RNG`` (the project's own wrapper) or a raw
``numpy.random.Generator`` and an :class:`InnovationTracker` for
structural changes.

Supported mutations
-------------------
- **mutate_weights** — perturb or replace connection weights.
- **add_connection** — insert a new enabled connection between two
  previously unconnected nodes.
- **add_node** — split an existing enabled connection into two,
  inserting a new hidden node in between.
- **toggle_connection** — enable/disable a random connection.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from thermal_inference_lab.neat.genome import (
    ConnectionGene,
    Genome,
    InnovationTracker,
    NodeGene,
)


# -------------------------------------------------------------- weights

def mutate_weights(
    genome: Genome,
    rng: np.random.Generator,
    *,
    perturb_rate: float = 0.8,
    perturb_std: float = 0.5,
    replace_rate: float = 0.1,
    weight_range: float = 2.0,
) -> None:
    """Perturb or replace every connection weight with some probability."""
    for conn in genome.connections.values():
        r = rng.random()
        if r < perturb_rate:
            conn.weight += float(rng.normal(0.0, perturb_std))
            conn.weight = float(np.clip(conn.weight, -weight_range, weight_range))
        elif r < perturb_rate + replace_rate:
            conn.weight = float(rng.uniform(-weight_range, weight_range))


# -------------------------------------------------------- add connection

def add_connection(
    genome: Genome,
    tracker: InnovationTracker,
    rng: np.random.Generator,
    *,
    weight_range: float = 1.0,
    max_attempts: int = 20,
) -> bool:
    """Add a new connection between two previously unconnected nodes.

    Returns ``True`` if a connection was added, ``False`` if no valid
    pair could be found after *max_attempts* random tries.
    """
    node_ids = list(genome.nodes.keys())
    if len(node_ids) < 2:
        return False
    existing = {(c.in_node, c.out_node) for c in genome.connections.values()}
    output_ids = set(genome.output_nodes)
    input_ids = set(genome.input_nodes) | set(genome.bias_nodes)

    for _ in range(max_attempts):
        src = int(rng.choice(node_ids))
        dst = int(rng.choice(node_ids))
        if src == dst:
            continue
        # Don't connect output -> output or input -> input.
        if src in output_ids and dst in output_ids:
            continue
        if dst in input_ids:
            continue  # inputs/bias can't receive connections
        if (src, dst) in existing:
            continue
        innov = tracker.get_innovation(src, dst)
        genome.connections[innov] = ConnectionGene(
            innovation=innov,
            in_node=src,
            out_node=dst,
            weight=float(rng.uniform(-weight_range, weight_range)),
            enabled=True,
        )
        return True
    return False


# ------------------------------------------------------------- add node

def add_node(
    genome: Genome,
    tracker: InnovationTracker,
    rng: np.random.Generator,
    *,
    next_node_id: Optional[int] = None,
) -> bool:
    """Split an enabled connection, inserting a new hidden node.

    The old connection is disabled.  Two new connections are created:
    ``in_node -> new_node`` (weight 1.0) and ``new_node -> out_node``
    (old weight).  Returns ``True`` on success.
    """
    enabled = [c for c in genome.connections.values() if c.enabled]
    if not enabled:
        return False
    conn = enabled[int(rng.integers(len(enabled)))]
    conn.enabled = False

    new_id = next_node_id if next_node_id is not None else (genome.max_node_id + 1)
    genome.nodes[new_id] = NodeGene(id=new_id, type="hidden", activation="sigmoid")

    innov_a = tracker.get_innovation(conn.in_node, new_id)
    genome.connections[innov_a] = ConnectionGene(
        innovation=innov_a,
        in_node=conn.in_node,
        out_node=new_id,
        weight=1.0,
        enabled=True,
    )
    innov_b = tracker.get_innovation(new_id, conn.out_node)
    genome.connections[innov_b] = ConnectionGene(
        innovation=innov_b,
        in_node=new_id,
        out_node=conn.out_node,
        weight=conn.weight,
        enabled=True,
    )
    return True


# -------------------------------------------------------- toggle enable

def toggle_connection(genome: Genome, rng: np.random.Generator) -> None:
    """Randomly toggle the enabled status of one connection."""
    conns = list(genome.connections.values())
    if not conns:
        return
    c = conns[int(rng.integers(len(conns)))]
    c.enabled = not c.enabled


# --------------------------------------------------- composite mutation

def mutate(
    genome: Genome,
    tracker: InnovationTracker,
    rng: np.random.Generator,
    *,
    weight_mutation_rate: float = 0.8,
    add_connection_rate: float = 0.05,
    add_node_rate: float = 0.03,
    toggle_rate: float = 0.02,
    perturb_std: float = 0.5,
    weight_range: float = 2.0,
) -> None:
    """Apply a battery of NEAT mutations to *genome* in-place."""
    if rng.random() < weight_mutation_rate:
        mutate_weights(genome, rng, perturb_std=perturb_std, weight_range=weight_range)
    if rng.random() < add_connection_rate:
        add_connection(genome, tracker, rng, weight_range=weight_range)
    if rng.random() < add_node_rate:
        add_node(genome, tracker, rng)
    if rng.random() < toggle_rate:
        toggle_connection(genome, rng)


__all__ = [
    "mutate_weights",
    "add_connection",
    "add_node",
    "toggle_connection",
    "mutate",
]