"""Base class for parameter-dict optimizers.

Optimizers operate on a *dictionary* ``{name: np.ndarray}`` of parameters
and consume a matching dictionary of gradients. This format lines up
naturally with :class:`RBMEnergy.params` and with the inference
routines, which build small parameter dicts on the fly.

Updates are performed *in place* on the parameter arrays so that any
object holding a reference to them (e.g. ``RBMEnergy.params``) sees
the new values immediately.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from thermal_inference_lab.core.validation import require_non_negative, require_positive


Params = dict[str, np.ndarray]


class Optimizer(ABC):
    """Abstract base optimizer."""

    def __init__(self, lr: float) -> None:
        require_positive(lr, name="lr")
        self.lr = float(lr)
        self.step_count: int = 0

    @abstractmethod
    def step(self, params: Params, grads: Params) -> None:
        """In-place update of ``params`` using ``grads``."""

    @staticmethod
    def _check_keys(params: Params, grads: Params) -> None:
        if set(params.keys()) != set(grads.keys()):
            missing = set(params) - set(grads)
            extra = set(grads) - set(params)
            raise KeyError(
                f"params/grads key mismatch: missing={missing}, extra={extra}"
            )
        for k in params:
            if params[k].shape != grads[k].shape:
                raise ValueError(
                    f"shape mismatch for '{k}': params {params[k].shape} vs grads {grads[k].shape}"
                )


__all__ = ["Optimizer", "Params"]
