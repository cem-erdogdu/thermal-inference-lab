"""Container for results returned by inference routines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FitResult:
    """Generic fit result with parameter dict + loss history + metadata."""

    params: dict[str, Any]
    loss_history: np.ndarray
    converged: bool
    n_iterations: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_loss(self) -> float:
        return float(self.loss_history[-1]) if self.loss_history.size else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "loss_history": self.loss_history.tolist(),
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "metadata": self.metadata,
        }


__all__ = ["FitResult"]
