"""Adam optimizer (Kingma & Ba, 2015) implemented in pure NumPy.

    m <- beta1 * m + (1 - beta1) * g
    v <- beta2 * v + (1 - beta2) * g**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    θ <- θ - lr * m_hat / (sqrt(v_hat) + eps)
"""

from __future__ import annotations

import numpy as np

from thermal_inference_lab.core.validation import require_positive
from thermal_inference_lab.optimizers.base import Optimizer, Params


class Adam(Optimizer):
    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(lr)
        require_positive(eps, name="eps")
        if not (0.0 <= beta1 < 1.0):
            raise ValueError("beta1 must be in [0, 1)")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError("beta2 must be in [0, 1)")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self._m: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}

    def step(self, params: Params, grads: Params) -> None:
        self._check_keys(params, grads)
        self.step_count += 1
        t = self.step_count
        bc1 = 1.0 - self.beta1 ** t
        bc2 = 1.0 - self.beta2 ** t
        for k, p in params.items():
            g = grads[k]
            m = self._m.setdefault(k, np.zeros_like(p))
            v = self._v.setdefault(k, np.zeros_like(p))
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            m_hat = m / bc1
            v_hat = v / bc2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


__all__ = ["Adam"]
