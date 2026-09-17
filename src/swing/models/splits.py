"""Time-forward splits, shared by every training script.

⚠️ There is deliberately NO shuffle option anywhere in this module, and no
`random_state`. agent-plan.md Step 5.3: "Use strict walk-forward splits. Train on
pairs from before date D, evaluate on swings after D. Never shuffle. Shuffled
splits on time-series data leak future information and will make a broken model
look excellent."

That failure is silent and flattering, which is the worst combination: a model
that has seen tomorrow's news scores brilliantly in evaluation and is worthless
in production. Every split here asserts that the last training timestamp is
strictly before the first evaluation one, so leakage raises instead of printing
a good number.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

__all__ = ["Split", "assert_time_forward", "expanding_folds", "split_at"]


def _when(row: Any, key: str) -> datetime | date:
    value = row[key] if isinstance(row, dict) else getattr(row, key)
    if value is None:
        raise ValueError(f"row has no {key}; a split cannot be time-forward without it")
    return value


@dataclass(frozen=True, slots=True)
class Split:
    train: list[Any]
    test: list[Any]
    cutoff: datetime | date

    def __repr__(self) -> str:      # keeps training logs readable
        return f"Split(train={len(self.train)}, test={len(self.test)}, cutoff={self.cutoff})"


def assert_time_forward(train: Sequence[Any], test: Sequence[Any], key: str) -> None:
    """Raise unless every training row predates every evaluation row."""
    if not train or not test:
        return
    newest_train = max(_when(r, key) for r in train)
    oldest_test = min(_when(r, key) for r in test)
    if newest_train >= oldest_test:
        raise ValueError(
            f"leakage: newest training row ({newest_train}) is not before the oldest "
            f"evaluation row ({oldest_test}). Splits must be strictly time-forward."
        )


def split_at[T](rows: Sequence[T], cutoff: datetime | date,
                key: str = "published_at") -> Split:
    """Everything before `cutoff` trains; everything on or after it evaluates."""
    train = [r for r in rows if _when(r, key) < cutoff]
    test = [r for r in rows if _when(r, key) >= cutoff]
    assert_time_forward(train, test, key)
    return Split(train=train, test=test, cutoff=cutoff)


def expanding_folds[T](rows: Sequence[T], n_folds: int = 3,
                       key: str = "published_at") -> list[Split]:
    """Expanding-window folds: train on the past, test on the next slice.

    Fold k trains on the first (k+1)/(n+1) of the timeline and evaluates on the
    slice after it, so the training window only ever grows. This is the honest
    analogue of k-fold for a time series — ordinary k-fold would put future rows
    in the training set of early folds.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    ordered = sorted(rows, key=lambda r: _when(r, key))
    if len(ordered) < n_folds + 1:
        raise ValueError(f"need at least {n_folds + 1} rows for {n_folds} folds, got {len(ordered)}")

    folds: list[Split] = []
    step = len(ordered) // (n_folds + 1)
    for k in range(1, n_folds + 1):
        boundary = step * k
        # Do not split inside a run of identical timestamps: rows sharing the
        # cutoff instant would straddle it and trip the leakage assertion.
        cutoff = _when(ordered[boundary], key)
        while boundary < len(ordered) and _when(ordered[boundary], key) == cutoff:
            boundary += 1
        if boundary >= len(ordered):
            continue
        train, test = ordered[:boundary], ordered[boundary:]
        assert_time_forward(train, test, key)
        folds.append(Split(train=train, test=test, cutoff=_when(ordered[boundary], key)))
    return folds
