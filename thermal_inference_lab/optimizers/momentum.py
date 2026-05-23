"""Classical (heavy-ball) momentum.

    v <- mu * v + g
    θ <- θ - lr * v
"""

from __future__ import annotations

import numpy as np

from thermal_inference_lab.core.validation import require_non_negative
from thermal_inference_lab.optimizers.base import Optimizer, Params


class Momentum(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        super().__init__(lr)
        require_non_negative(momentum, name="momentum")
        self.momentum = float(momentum)
        self._velocity: dict[str, np.ndarray] = {}

    def step(self, params: Params, grads: Params) -> None:
        self._check_keys(params, grads)
        for k, p in params.items():
            v = self._velocity.setdefault(k, np.zeros_like(p))
            v *= self.momentum
            v += grads[k]
            p -= self.lr * v
        self.step_count += 1


__all__ = ["Momentum"]
