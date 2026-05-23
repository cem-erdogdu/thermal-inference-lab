"""Numerically stable helper math used in samplers and inference.

Functions here intentionally avoid scipy: we re-derive the small set of
primitives we need (log-sum-exp, sigmoid, softplus, etc.) so that the
package stays NumPy-only.
"""

from __future__ import annotations

import numpy as np

from thermal_inference_lab.core.constants import EPS, LOG_EPS


def logsumexp(x: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    """Numerically stable ``log(sum(exp(x)))``.

    Equivalent to ``scipy.special.logsumexp`` for finite inputs. Handles
    -inf entries by treating them as contributing zero to the sum.
    """
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    # Replace -inf max with 0 so we don't get NaNs from inf - inf below.
    m = np.where(np.isfinite(m), m, 0.0)
    shifted = np.exp(x - m)
    s = np.sum(shifted, axis=axis, keepdims=True)
    out = np.log(s + EPS) + m
    if not keepdims:
        out = np.squeeze(out, axis=axis) if axis is not None else out.reshape(())
    return out


def safe_log(x: np.ndarray | float) -> np.ndarray:
    """``log(x)`` with a floor to avoid -inf for x=0."""
    return np.log(np.maximum(np.asarray(x, dtype=float), np.exp(LOG_EPS)))


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    x = np.asarray(x, dtype=float)
    pos = x >= 0
    out = np.empty_like(x)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    z = np.exp(x[~pos])
    out[~pos] = z / (1.0 + z)
    return out


def softplus(x: np.ndarray | float) -> np.ndarray:
    """``log(1 + exp(x))`` without overflow."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 30.0, x, np.log1p(np.exp(np.minimum(x, 30.0))))


def standardize(x: np.ndarray, axis: int | None = None, eps: float = EPS) -> np.ndarray:
    """Zero-mean, unit-variance normalisation along ``axis``."""
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    return (x - mean) / (std + eps)


def running_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Centred running mean along axis 0."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if window == 1:
        return x.copy()
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def autocorrelation(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Sample autocorrelation function up to ``max_lag``.

    Uses the FFT for O(N log N) cost. Returns an array of length
    ``max_lag + 1`` normalised so that the value at lag 0 is 1.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if max_lag is None:
        max_lag = n // 2
    max_lag = min(max_lag, n - 1)
    x = x - x.mean()
    # Pad to next power of two for FFT speed.
    size = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, n=size)
    acf = np.fft.irfft(f * np.conj(f), n=size)[:max_lag + 1]
    acf /= acf[0] + EPS
    return acf


def integrated_autocorrelation_time(acf: np.ndarray, c: float = 5.0) -> float:
    """Sokal's automatic windowing estimator for tau_int.

    ``acf`` is the normalised autocorrelation function (lag 0 == 1).
    The window M is the smallest M such that M >= c * tau(M).
    """
    if acf.size == 0:
        return float("nan")
    tau = 0.5
    for m in range(1, acf.size):
        tau += acf[m]
        if m >= c * (2.0 * tau):
            return float(2.0 * tau)
    return float(2.0 * tau)


__all__ = [
    "logsumexp",
    "safe_log",
    "sigmoid",
    "softplus",
    "standardize",
    "running_mean",
    "autocorrelation",
    "integrated_autocorrelation_time",
]
