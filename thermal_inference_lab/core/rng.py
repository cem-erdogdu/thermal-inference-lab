"""Reproducible random-number-generation wrapper.

Every stochastic routine in the library accepts an :class:`RNG` so that
runs are bit-for-bit reproducible from a single integer seed. We wrap
``numpy.random.Generator`` rather than the legacy ``np.random.*`` global
state, which is essential for parallel/sub-stream sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class RNG:
    """Thin reproducible wrapper around ``numpy.random.Generator``.

    Construct from a seed (``RNG(123)``) or from an existing generator
    (``RNG.from_generator(g)``). Use :meth:`spawn` to derive independent
    sub-streams; this is the recommended pattern for parallel samplers.
    """

    seed: int
    _generator: np.random.Generator | None = None

    def __post_init__(self) -> None:
        if self._generator is None:
            self._generator = np.random.default_rng(int(self.seed))

    # ------------------------------------------------------------------ ctors
    @classmethod
    def from_generator(cls, generator: np.random.Generator, seed: int = 0) -> "RNG":
        return cls(seed=seed, _generator=generator)

    # ------------------------------------------------------------------ core
    @property
    def generator(self) -> np.random.Generator:
        assert self._generator is not None
        return self._generator

    def uniform(self, low: float = 0.0, high: float = 1.0, size=None) -> np.ndarray:
        return self.generator.uniform(low, high, size=size)

    def normal(self, loc: float = 0.0, scale: float = 1.0, size=None) -> np.ndarray:
        return self.generator.normal(loc, scale, size=size)

    def integers(self, low: int, high: int | None = None, size=None) -> np.ndarray:
        return self.generator.integers(low, high, size=size)

    def choice(self, a, size=None, replace: bool = True, p=None):
        return self.generator.choice(a, size=size, replace=replace, p=p)

    def random(self, size=None) -> np.ndarray:
        return self.generator.random(size=size)

    def permutation(self, x):
        return self.generator.permutation(x)

    def shuffle(self, x) -> None:
        self.generator.shuffle(x)

    # ------------------------------------------------------------------ spawn
    def spawn(self, n: int) -> list["RNG"]:
        """Return ``n`` independent sub-streams using SeedSequence spawning."""
        if n < 1:
            raise ValueError("spawn count must be >= 1")
        ss = np.random.SeedSequence(int(self.seed))
        children = ss.spawn(n)
        return [
            RNG(seed=int(child.entropy) & 0xFFFFFFFF, _generator=np.random.default_rng(child))
            for child in children
        ]


def split_rng(rng: RNG, names: Sequence[str]) -> dict[str, RNG]:
    """Convenience helper to label spawned sub-streams."""
    streams = rng.spawn(len(names))
    return dict(zip(names, streams))


def as_rng(rng_or_seed: RNG | int | None, default_seed: int = 0) -> RNG:
    """Coerce user input into an :class:`RNG` instance."""
    if rng_or_seed is None:
        return RNG(default_seed)
    if isinstance(rng_or_seed, RNG):
        return rng_or_seed
    return RNG(int(rng_or_seed))


__all__ = ["RNG", "split_rng", "as_rng"]
