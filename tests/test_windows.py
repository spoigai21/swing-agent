"""Retrieval-window boundaries. The pre/post split is the system's honesty."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from swing.analysis.windows import windows_for

ONSET = datetime(2026, 8, 27, 14, 5, tzinfo=UTC)
PREV_CLOSE = datetime(2026, 8, 26, 20, 0, tzinfo=UTC)
OPEN = datetime(2026, 8, 27, 13, 30, tzinfo=UTC)


@pytest.mark.parametrize("swing_type", ["gap", "intraday", "mixed", "drift", "unknown"])
def test_pre_and_post_never_overlap(swing_type):
    pre, post = windows_for(swing_type, ONSET, PREV_CLOSE, OPEN, PREV_CLOSE)
    assert pre.end <= post.start, f"{swing_type}: windows overlap"


@pytest.mark.parametrize("swing_type", ["gap", "intraday", "mixed", "drift", "unknown"])
def test_windows_are_non_empty_and_ordered(swing_type):
    pre, post = windows_for(swing_type, ONSET, PREV_CLOSE, OPEN, PREV_CLOSE)
    assert pre.start < pre.end
    assert post.start < post.end


def test_gap_window_is_overnight_only():
    # For a gap move the catalyst arrived overnight, so a narrow window raises
    # precision at no cost to recall.
    pre, _ = windows_for("gap", ONSET, PREV_CLOSE, OPEN)
    assert pre.start == PREV_CLOSE and pre.end == OPEN
    assert pre.end - pre.start < timedelta(hours=24)


def test_intraday_window_ends_at_onset_not_the_close():
    pre, post = windows_for("intraday", ONSET, PREV_CLOSE)
    assert pre.end == ONSET
    assert post.start == ONSET


def test_article_after_onset_is_post_move():
    pre, post = windows_for("intraday", ONSET, PREV_CLOSE)
    after = ONSET + timedelta(minutes=1)
    assert not pre.contains(after)
    assert post.contains(after)


def test_article_before_onset_is_pre_move():
    pre, post = windows_for("intraday", ONSET, PREV_CLOSE)
    before = ONSET - timedelta(hours=2)
    assert pre.contains(before)
    assert not post.contains(before)


def test_unknown_type_widens_to_the_fallback_window():
    pre, _ = windows_for("unknown", ONSET, PREV_CLOSE)
    assert pre.end - pre.start == timedelta(hours=48)


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError, match="naive"):
        windows_for("intraday", ONSET.replace(tzinfo=None), PREV_CLOSE)


def test_bad_swing_type_raises():
    with pytest.raises(ValueError, match="unknown swing_type"):
        windows_for("sideways", ONSET, PREV_CLOSE)
