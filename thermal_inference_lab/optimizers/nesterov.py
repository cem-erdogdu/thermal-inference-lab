"""Nesterov-accelerated gradient descent.

We use the "look-ahead" form

    v_new = mu * v + g
    θ_new = θ - lr * (mu * v_new + g)

which is equivalent to evaluating the gradient at θ + mu * v in the
classical formulation.
"""

from __future__ import annotations

import numpy as np

from thermal_inference_lab.core.validation import require_non_negative
from thermal_inference_lab.optimizers.base import Optimizer, Params


class Nesterov(Optimizer):
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
            p -= self.lr * (self.momentum * v + grads[k])
        self.step_count += 1


__all__ = ["Nesterov"]
