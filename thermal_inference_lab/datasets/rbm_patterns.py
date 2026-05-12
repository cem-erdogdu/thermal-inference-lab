"""Classic binary pattern datasets used to test RBM training."""

from __future__ import annotations

import itertools

import numpy as np

from thermal_inference_lab.core.rng import as_rng


def bars_and_stripes(size: int = 4) -> np.ndarray:
    """All bars-and-stripes patterns for a ``size x size`` grid.

    A bars-and-stripes config has every row either all-0 or all-1, OR
    every column either all-0 or all-1. The dataset has 2*2^size - 2
    unique patterns (subtracting the two solid colours counted twice).
    """
    patterns = []
    for bits in itertools.product([0, 1], repeat=size):
        img = np.tile(np.asarray(bits, dtype=np.int8)[:, None], (1, size))
        patterns.append(img.flatten())
    for bits in itertools.product([0, 1], repeat=size):
        img = np.tile(np.asarray(bits, dtype=np.int8)[None, :], (size, 1))
        patterns.append(img.flatten())
    patterns = np.unique(np.asarray(patterns), axis=0).astype(np.int8)
    return patterns


def block_patterns(size: int = 4, seed: int = 0, n_per_class: int = 50) -> np.ndarray:
    """Two-class block patterns: top half on vs bottom half on, with noise."""
    rng = as_rng(seed)
    top = np.zeros((n_per_class, size, size), dtype=np.int8)
    bot = np.zeros((n_per_class, size, size), dtype=np.int8)
    top[:, : size // 2, :] = 1
    bot[:, size // 2 :, :] = 1
    data = np.concatenate([top, bot], axis=0).reshape(-1, size * size)
    flips = rng.random(data.shape) < 0.05
    data = np.where(flips, 1 - data, data).astype(np.int8)
    rng.shuffle(data)
    return data


__all__ = ["bars_and_stripes", "block_patterns"]
