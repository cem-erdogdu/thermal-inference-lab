"""NEAT speciation — compatibility distance and species management.

Genomes are grouped into species based on a compatibility distance
threshold.  The distance measures structural dissimilarity (disjoint
and excess genes) plus weight divergence among matching genes.

Species protect innovation by ensuring that novel topologies compete
primarily with similar topologies (explicit fitness sharing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from thermal_inference_lab.neat.genome import Genome


# --------------------------------------------------- compatibility distance

def compatibility_distance(
    g1: Genome,
    g2: Genome,
    *,
    c1: float = 1.0,
    c2: float = 1.0,
    c3: float = 0.4,
) -> float:
    """Compute the NEAT compatibility distance between two genomes.

    .. math::

        \\delta = \\frac{c_1 E}{N} + \\frac{c_2 D}{N} + c_3 \\bar{W}

    where *E* = excess genes, *D* = disjoint genes, *N* = max genome
    size (at least 1), and :math:`\\bar{W}` = average weight difference
    of matching genes.
    """
    innovs1 = set(g1.connections.keys())
    innovs2 = set(g2.connections.keys())
    if not innovs1 and not innovs2:
        return 0.0

    matching = innovs1 & innovs2
    all_innovs = innovs1 | innovs2

    # Separate disjoint and excess genes.
    max1 = max(innovs1) if innovs1 else -1
    max2 = max(innovs2) if innovs2 else -1
    threshold = min(max1, max2)

    excess = 0
    disjoint = 0
    for innov in all_innovs - matching:
        if innov > threshold:
            excess += 1
        else:
            disjoint += 1

    # Average weight difference of matching genes.
    if matching:
        weight_diff = sum(
            abs(g1.connections[i].weight - g2.connections[i].weight)
            for i in matching
        ) / len(matching)
    else:
        weight_diff = 0.0

    n = max(len(innovs1), len(innovs2), 1)
    return (c1 * excess / n) + (c2 * disjoint / n) + (c3 * weight_diff)


# --------------------------------------------------------------- species

@dataclass
class Species:
    """A group of genomes with similar topology."""
    id: int
    representative: Genome
    members: List[Genome] = field(default_factory=list)
    best_fitness: float = 0.0
    stagnation: int = 0
    age: int = 0


class SpeciationManager:
    """Assign genomes to species using compatibility distance.

    Parameters
    ----------
    threshold : float
        Compatibility distance threshold for belonging to a species.
    c1, c2, c3 : float
        Coefficients for excess, disjoint, and weight distance.
    max_stagnation : int
        Species that don't improve for this many generations are removed
        (except the top two).
    """

    def __init__(
        self,
        threshold: float = 3.0,
        c1: float = 1.0,
        c2: float = 1.0,
        c3: float = 0.4,
        max_stagnation: int = 15,
    ) -> None:
        self.threshold = threshold
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.max_stagnation = max_stagnation
        self._species: Dict[int, Species] = {}
        self._next_species_id: int = 0

    @property
    def species(self) -> Dict[int, Species]:
        return self._species

    @property
    def num_species(self) -> int:
        return len(self._species)

    def speciate(self, population: List[Genome], rng: np.random.Generator) -> None:
        """Assign every genome in *population* to a species."""
        # Reset members but keep representatives.
        for sp in self._species.values():
            sp.members.clear()

        for genome in population:
            placed = False
            for sp in self._species.values():
                d = compatibility_distance(
                    genome, sp.representative,
                    c1=self.c1, c2=self.c2, c3=self.c3,
                )
                if d < self.threshold:
                    sp.members.append(genome)
                    genome.species_id = sp.id
                    placed = True
                    break
            if not placed:
                sid = self._next_species_id
                self._next_species_id += 1
                new_sp = Species(id=sid, representative=genome, members=[genome])
                self._species[sid] = new_sp
                genome.species_id = sid

        # Remove empty species.
        empty = [sid for sid, sp in self._species.items() if not sp.members]
        for sid in empty:
            del self._species[sid]

        # Update representatives and stagnation.
        for sp in self._species.values():
            sp.age += 1
            best = max(sp.members, key=lambda g: g.fitness)
            if best.fitness > sp.best_fitness:
                sp.best_fitness = best.fitness
                sp.stagnation = 0
            else:
                sp.stagnation += 1
            # New representative is a random member.
            sp.representative = sp.members[int(rng.integers(len(sp.members)))]

    def remove_stagnant(self, *, keep_top: int = 2) -> List[int]:
        """Remove stagnant species, keeping the top *keep_top* by best fitness.

        Returns the ids of removed species.
        """
        if len(self._species) <= keep_top:
            return []
        # Sort species by best_fitness descending.
        ranked = sorted(
            self._species.values(),
            key=lambda sp: sp.best_fitness,
            reverse=True,
        )
        protected = {sp.id for sp in ranked[:keep_top]}
        removed: List[int] = []
        for sp in ranked[keep_top:]:
            if sp.stagnation >= self.max_stagnation:
                removed.append(sp.id)
        for sid in removed:
            del self._species[sid]
        return removed

    def adjusted_fitness(self) -> Dict[int, float]:
        """Return explicit fitness sharing: fitness / species_size."""
        result: Dict[int, float] = {}
        for sp in self._species.values():
            n = len(sp.members)
            for g in sp.members:
                result[g.genome_id] = g.fitness / n if n > 0 else 0.0
        return result


__all__ = [
    "compatibility_distance",
    "Species",
    "SpeciationManager",
]