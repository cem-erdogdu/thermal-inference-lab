"""Lightweight serialization helpers using JSON + ``.npz``.

We deliberately avoid pickle: it is unsafe to load untrusted files and
hides shape/dtype changes during refactors. JSON for metadata + NumPy
``.npz`` for arrays is sufficient for everything in this repo.
"""

from __future__ import annotations

import json
import os
from dataclasses import is_dataclass, asdict
from typing import Any, Mapping

import numpy as np


def _coerce_for_json(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Mapping):
        return {str(k): _coerce_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_coerce_for_json(v) for v in obj]
    if is_dataclass(obj):
        return _coerce_for_json(asdict(obj))
    return repr(obj)


def save_json(path: str, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` as pretty-printed JSON, creating parent dirs."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_coerce_for_json(payload), fh, indent=2, sort_keys=True)


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_npz(path: str, **arrays: np.ndarray) -> None:
    """Save a collection of named arrays to a compressed ``.npz`` file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


__all__ = ["save_json", "load_json", "save_npz", "load_npz"]
