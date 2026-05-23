"""NEAT crossover operator.

Crossover aligns two parent genomes by innovation number.  Genes that
appear in both parents (matching genes) are inherited randomly from
either parent.  Disjoint and excess genes are inherited from the
fitter parent only.  If both parents have equal fitness, disjoint and
excess genes are inherited from both parents randomly.
"""

from __future__ import annotations

import copy
from typing import Dict

import numpy as np

from thermal_inference_lab.neat.genome import (
    ConnectionGene,
    Genome,
    NodeGene,
)


def crossover(
    parent1: Genome,
    parent2: Genome,
    rng: np.random.Generator,
    *,
    disabled_gene_inherit_rate: float = 0.75,
) -> Genome:
    """Produce an offspring genome by NEAT crossover.

    *parent1* is assumed to be the fitter (or equal) parent.  If the
    caller doesn't know which is fitter, swap so that
    ``parent1.fitness >= parent2.fitness`` before calling.

    Parameters
    ----------
    disabled_gene_inherit_rate : float
        Probability that a gene disabled in either parent stays disabled
        in the child.
    """
    child = Genome()

    # Determine which parent is fitter.
    if parent1.fitness >= parent2.fitness:
        fitter, other = parent1, parent2
    else:
        fitter, other = parent2, parent1

    equal_fitness = abs(parent1.fitness - parent2.fitness) < 1e-12

    # ---- inherit connection genes ----
    all_innovations = set(fitter.connections.keys()) | set(other.connections.keys())
    for innov in sorted(all_innovations):
        in_fitter = innov in fitter.connections
        in_other = innov in other.connections
        if in_fitter and in_other:
            # Matching gene — pick randomly.
            if rng.random() < 0.5:
                gene = copy.copy(fitter.connections[innov])
            else:
                gene = copy.copy(other.connections[innov])
            # Re-enable check.
            if (not fitter.connections[innov].enabled or
                    not other.connections[innov].enabled):
                if rng.random() < disabled_gene_inherit_rate:
                    gene.enabled = False
                else:
                    gene.enabled = True
        elif in_fitter:
            gene = copy.copy(fitter.connections[innov])
        elif equal_fitness:
            # Both parents contribute disjoint/excess when fitness is equal.
            gene = copy.copy(other.connections[innov])
        else:
            continue  # Disjoint/excess from weaker parent — skip.
        child.connections[innov] = gene

    # ---- inherit node genes ----
    needed_nodes: set = set()
    for c in child.connections.values():
        needed_nodes.add(c.in_node)
        needed_nodes.add(c.out_node)
    # Always include input/bias/output nodes from fitter parent.
    for nid, node in fitter.nodes.items():
        if node.type in ("input", "bias", "output"):
            needed_nodes.add(nid)

    for nid in sorted(needed_nodes):
        if nid in fitter.nodes:
            child.nodes[nid] = copy.copy(fitter.nodes[nid])
        elif nid in other.nodes:
            child.nodes[nid] = copy.copy(other.nodes[nid])
        else:
            # Safety fallback — hidden node must exist somewhere.
            child.nodes[nid] = NodeGene(id=nid, type="hidden", activation="sigmoid")

    return child


__all__ = ["crossover"]