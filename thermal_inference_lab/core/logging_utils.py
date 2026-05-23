"""Lightweight, dependency-free logging helpers.

We intentionally do not pull in any logging frameworks; experiments
print progress at a configurable interval and tests can easily silence
output by passing ``verbose=False``.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Iterator


class ProgressReporter:
    """Periodically prints progress messages with elapsed time.

    Parameters
    ----------
    total : total number of steps expected (used to compute %).
    every : print every ``every`` steps (0 disables periodic prints).
    label : prefix for each log line.
    stream : output stream (defaults to ``sys.stdout``).
    enabled : easy global on/off switch.
    """

    def __init__(
        self,
        total: int,
        every: int = 0,
        label: str = "progress",
        stream=None,
        enabled: bool = True,
    ) -> None:
        self.total = max(int(total), 1)
        self.every = int(every)
        self.label = label
        self.stream = stream or sys.stdout
        self.enabled = enabled
        self._start = time.time()
        self._last_print: int = -1

    def update(self, step: int, extra: str = "") -> None:
        if not self.enabled or self.every <= 0:
            return
        if step == self._last_print:
            return
        if step % self.every != 0 and step != self.total - 1:
            return
        elapsed = time.time() - self._start
        pct = 100.0 * (step + 1) / self.total
        msg = f"[{self.label}] step={step + 1}/{self.total} ({pct:5.1f}%) elapsed={elapsed:6.2f}s"
        if extra:
            msg = f"{msg} | {extra}"
        print(msg, file=self.stream, flush=True)
        self._last_print = step


@contextmanager
def timed_block(label: str, enabled: bool = True, stream=None) -> Iterator[None]:
    """Context manager that prints elapsed time for a code block."""
    stream = stream or sys.stdout
    start = time.time()
    yield
    if enabled:
        print(f"[{label}] finished in {time.time() - start:.3f}s", file=stream, flush=True)


__all__ = ["ProgressReporter", "timed_block"]
