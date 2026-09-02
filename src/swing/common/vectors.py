"""Coerce a stored embedding into a numpy array.

pgvector returns a `Vector` object once types are registered, and its text form
otherwise. Both appear in practice, so every consumer goes through here.
"""
from __future__ import annotations

import numpy as np


def as_array(v) -> np.ndarray:
    if v is None:
        return np.zeros(0, dtype=float)
    if isinstance(v, np.ndarray):
        return v.astype(float)
    if hasattr(v, "to_numpy"):
        return np.asarray(v.to_numpy(), dtype=float)
    if isinstance(v, str):
        return np.array([float(x) for x in v.strip("[]").split(",")], dtype=float)
    return np.asarray(list(v), dtype=float)
