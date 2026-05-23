"""Feed-forward network execution from a NEAT genome.

The network is built by topologically sorting the enabled connections
and propagating activations from inputs through hidden layers to outputs.
No external graph library is used — the sort is a simple Kahn's algorithm
over the nodes reachable via enabled connections.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from thermal_inference_lab.neat.genome import Genome, NodeGene

# ------------------------------------------------------------- activations

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(x, -60.0, 60.0))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _identity(x: np.ndarray) -> np.ndarray:
    return x


_ACTIVATIONS = {
    "sigmoid": _sigmoid,
    "tanh": _tanh,
    "relu": _relu,
    "identity": _identity,
}


def get_activation(name: str):
    """Return the activation callable for *name*."""
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name!r}")
    return _ACTIVATIONS[name]


# --------------------------------------------------------- topological sort

def _topological_order(genome: Genome) -> List[int]:
    """Return node ids in feed-forward evaluation order.

    Input and bias nodes come first (they have no in-edges from the
    network's perspective).  Then hidden nodes in dependency order.
    Output nodes come last.
    """
    # Build adjacency from enabled connections only.
    children: Dict[int, List[int]] = {}
    in_degree: Dict[int, int] = {}
    active_nodes: set = set()
    for c in genome.connections.values():
        if not c.enabled:
            continue
        active_nodes.add(c.in_node)
        active_nodes.add(c.out_node)
        children.setdefault(c.in_node, []).append(c.out_node)
        in_degree[c.out_node] = in_degree.get(c.out_node, 0) + 1
        in_degree.setdefault(c.in_node, in_degree.get(c.in_node, 0))

    # All nodes that appear in connections.
    # Add input/bias/output nodes even if unconnected.
    for nid, node in genome.nodes.items():
        if node.type in ("input", "bias", "output"):
            active_nodes.add(nid)
            in_degree.setdefault(nid, 0)

    # Kahn's algorithm.
    queue: List[int] = []
    for nid in sorted(active_nodes):
        if in_degree.get(nid, 0) == 0:
            queue.append(nid)

    order: List[int] = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for child in children.get(nid, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    # If there are cycles, some nodes are missing — append them at the end
    # (NEAT is nominally feed-forward but mutations can create cycles).
    remaining = active_nodes - set(order)
    order.extend(sorted(remaining))
    return order


# -------------------------------------------------------- network class

class FeedForwardNetwork:
    """A compiled feed-forward network from a NEAT genome.

    Build once with :meth:`from_genome`, then call :meth:`activate`
    repeatedly for each input vector.  This avoids re-sorting on
    every evaluation step.
    """

    __slots__ = ("_order", "_node_act", "_edges", "_input_ids",
                 "_output_ids", "_bias_ids")

    def __init__(
        self,
        order: List[int],
        node_act: Dict[int, str],
        edges: List[Tuple[int, int, float]],
        input_ids: List[int],
        output_ids: List[int],
        bias_ids: List[int],
    ) -> None:
        self._order = order
        self._node_act = node_act
        self._edges = edges  # (in_node, out_node, weight)
        self._input_ids = input_ids
        self._output_ids = output_ids
        self._bias_ids = bias_ids

    @classmethod
    def from_genome(cls, genome: Genome) -> "FeedForwardNetwork":
        order = _topological_order(genome)
        node_act = {nid: genome.nodes[nid].activation for nid in genome.nodes}
        edges = [
            (c.in_node, c.out_node, c.weight)
            for c in genome.connections.values()
            if c.enabled
        ]
        return cls(
            order=order,
            node_act=node_act,
            edges=edges,
            input_ids=genome.input_nodes,
            output_ids=genome.output_nodes,
            bias_ids=genome.bias_nodes,
        )

    def activate(self, inputs: np.ndarray) -> np.ndarray:
        """Propagate *inputs* (1-D array) and return output activations."""
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.shape[0] != len(self._input_ids):
            raise ValueError(
                f"Expected {len(self._input_ids)} inputs, got {inputs.shape[0]}"
            )
        values: Dict[int, float] = {}
        # Clamp inputs.
        for i, nid in enumerate(self._input_ids):
            values[nid] = float(inputs[i])
        # Bias nodes output 1.0.
        for nid in self._bias_ids:
            values[nid] = 1.0

        # Build incoming edges per node.
        incoming: Dict[int, List[Tuple[int, float]]] = {}
        for src, dst, w in self._edges:
            incoming.setdefault(dst, []).append((src, w))

        # Propagate in topological order.
        for nid in self._order:
            if nid in values:
                # Already set (input / bias); apply activation only if not
                # an input or bias node (those use identity regardless).
                node_type = self._node_act.get(nid, "identity")
                if nid not in self._input_ids and nid not in self._bias_ids:
                    values[nid] = float(get_activation(node_type)(np.array(values[nid])))
                continue
            # Aggregate incoming signals.
            agg = 0.0
            for src, w in incoming.get(nid, []):
                agg += values.get(src, 0.0) * w
            act_fn = get_activation(self._node_act.get(nid, "sigmoid"))
            values[nid] = float(act_fn(np.array(agg)))

        return np.array([values.get(nid, 0.0) for nid in self._output_ids])


__all__ = [
    "FeedForwardNetwork",
    "get_activation",
]