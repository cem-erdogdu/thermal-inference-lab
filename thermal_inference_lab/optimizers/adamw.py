"""AdamW (Loshchilov & Hutter, 2019).

Decouples L2 weight decay from the adaptive learning rate by applying
``- lr * wd * theta`` directly to the parameters instead of folding it
into the gradient.
"""

from __future__ import annotations

import numpy as np

from thermal_inference_lab.core.validation import require_non_negative
from thermal_inference_lab.optimizers.adam import Adam
from thermal_inference_lab.optimizers.base import Params


class AdamW(Adam):
    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        require_non_negative(weight_decay, name="weight_decay")
        self.weight_decay = float(weight_decay)

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
            if self.weight_decay > 0.0:
                p -= self.lr * self.weight_decay * p


__all__ = ["AdamW"]
