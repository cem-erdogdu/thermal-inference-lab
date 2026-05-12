"""Vanilla stochastic gradient descent."""

from __future__ import annotations

import numpy as np

from thermal_inference_lab.optimizers.base import Optimizer, Params


class SGD(Optimizer):
    """θ <- θ - lr * g."""

    def step(self, params: Params, grads: Params) -> None:
        self._check_keys(params, grads)
        for k, p in params.items():
            p -= self.lr * grads[k]
        self.step_count += 1


__all__ = ["SGD"]
