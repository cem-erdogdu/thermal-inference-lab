"""Restricted Boltzmann Machine energy.

This is the canonical Hinton-style RBM with binary {0,1} visible and
hidden units. The joint energy is

    E(v, h) = - v^T W h - a^T v - b^T h.

Marginalising over the hidden units gives the *free energy*

    F(v) = - a^T v - sum_j softplus(W_j^T v + b_j).

The conditional distributions are factorial Bernoullis:

    p(h_j = 1 | v) = sigmoid(W_j^T v + b_j)
    p(v_i = 1 | h) = sigmoid(W_i  h  + a_i)

The class wraps weights/biases as plain NumPy arrays so that Adam,
SGD and friends from the ``optimizers`` package can update them in
place via a parameter dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thermal_inference_lab.core.math_utils import sigmoid, softplus
from thermal_inference_lab.core.rng import RNG
from thermal_inference_lab.core.validation import require_binary_array, require_shape
from thermal_inference_lab.energy.base import EnergyModel


@dataclass
class RBMParameters:
    W: np.ndarray  # (n_visible, n_hidden)
    a: np.ndarray  # (n_visible,)
    b: np.ndarray  # (n_hidden,)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"W": self.W, "a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, d: dict[str, np.ndarray]) -> "RBMParameters":
        return cls(W=d["W"], a=d["a"], b=d["b"])


class RBMEnergy(EnergyModel):
    """RBM energy model. Visible/hidden units are {0,1}."""

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        rng: RNG | None = None,
        weight_scale: float = 0.01,
    ) -> None:
        if n_visible < 1 or n_hidden < 1:
            raise ValueError("n_visible and n_hidden must be >= 1")
        self.n_visible = int(n_visible)
        self.n_hidden = int(n_hidden)
        if rng is None:
            rng = RNG(0)
        self.params = RBMParameters(
            W=rng.normal(0.0, weight_scale, size=(self.n_visible, self.n_hidden)),
            a=np.zeros(self.n_visible),
            b=np.zeros(self.n_hidden),
        )

    # ----- EnergyModel API: visible-side view (free energy as "total") ----
    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self.n_visible,)

    @property
    def n_sites(self) -> int:
        return self.n_visible

    def random_state(self, rng: RNG) -> np.ndarray:
        return rng.integers(0, 2, size=self.n_visible).astype(np.int8)

    def total_energy(self, state: np.ndarray) -> float:
        """Returns the free energy F(v); used as "energy" for Boltzmann ops."""
        return float(self.free_energy(state))

    def local_energy_delta(self, state: np.ndarray, site: tuple[int, ...]) -> float:
        """ΔF when flipping a single visible bit."""
        require_binary_array(state, name="state")
        (i,) = site
        v = state.astype(np.float64)
        flipped = v.copy()
        flipped[i] = 1.0 - flipped[i]
        return float(self.free_energy(flipped) - self.free_energy(v))

    # ---------------------------------------------------------- core math
    def free_energy(self, v: np.ndarray) -> float:
        """F(v) = - a^T v - sum_j softplus(W_j^T v + b_j)."""
        require_shape(np.asarray(v), (self.n_visible,), name="v")
        v = v.astype(np.float64)
        Wv_b = v @ self.params.W + self.params.b
        return float(-(self.params.a @ v) - softplus(Wv_b).sum())

    def joint_energy(self, v: np.ndarray, h: np.ndarray) -> float:
        v = v.astype(np.float64)
        h = h.astype(np.float64)
        return float(-(v @ self.params.W @ h) - self.params.a @ v - self.params.b @ h)

    # ------------------------------------------------------ conditionals
    def prob_hidden_given_visible(self, v: np.ndarray) -> np.ndarray:
        v = np.atleast_2d(v.astype(np.float64))
        return sigmoid(v @ self.params.W + self.params.b)

    def prob_visible_given_hidden(self, h: np.ndarray) -> np.ndarray:
        h = np.atleast_2d(h.astype(np.float64))
        return sigmoid(h @ self.params.W.T + self.params.a)

    def sample_hidden_given_visible(self, v: np.ndarray, rng: RNG) -> np.ndarray:
        p = self.prob_hidden_given_visible(v)
        return (rng.random(p.shape) < p).astype(np.int8)

    def sample_visible_given_hidden(self, h: np.ndarray, rng: RNG) -> np.ndarray:
        p = self.prob_visible_given_hidden(h)
        return (rng.random(p.shape) < p).astype(np.int8)

    # ----------------------------- gradient of negative log-likelihood
    def free_energy_gradients(self, V: np.ndarray) -> dict[str, np.ndarray]:
        """Expectation of ∂F/∂θ taken over a *batch* of visible configs V.

        For an RBM:
            ∂F/∂a = - v
            ∂F/∂b = - sigmoid(W^T v + b)
            ∂F/∂W = - v outer sigmoid(W^T v + b)
        """
        V = np.atleast_2d(V.astype(np.float64))
        ph = sigmoid(V @ self.params.W + self.params.b)  # (B, n_hidden)
        grad_a = -V.mean(axis=0)
        grad_b = -ph.mean(axis=0)
        grad_W = -(V.T @ ph) / V.shape[0]
        return {"W": grad_W, "a": grad_a, "b": grad_b}


__all__ = ["RBMEnergy", "RBMParameters"]
