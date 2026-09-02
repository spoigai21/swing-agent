"""Onset classification on hand-built bar series.

This is the plan's second irrecoverable mistake: it passes every gate while
producing wrong answers, because the timestamps all look correct. So the tests
construct series whose answer is known by construction.

The pure classification logic is duplicated here rather than hitting the
database, so the maths is testable in isolation.
"""
from __future__ import annotations

import numpy as np
import pytest

GAP_DOMINANT = 0.7
GAP_MIXED = 0.3


def classify(prev_close: float, closes: list[float]) -> tuple[str, int, float]:
    """Return (swing_type, onset_index, gap_share) — mirrors analysis/onset.locate."""
    px = np.array(closes, dtype=float)
    ar = np.diff(np.log(px), prepend=np.log(prev_close))
    car = np.cumsum(ar)
    gap, total = float(ar[0]), float(car[-1])
    if total == 0:
        return "unknown", 0, 0.0
    share = abs(gap) / abs(total)
    if share > GAP_DOMINANT:
        return "gap", 0, share
    if share > GAP_MIXED:
        return "mixed", 0, share
    direction = np.sign(total - car[0])
    return "intraday", int(np.argmin(direction * car)), share


class TestClassification:
    def test_pure_gap(self):
        # Jumps overnight, then flat all session.
        t, idx, share = classify(100.0, [110.0] * 12)
        assert t == "gap" and idx == 0 and share > 0.95

    def test_pure_intraday_no_gap(self):
        # Opens unchanged, then trends down through the session.
        closes = [100.0] + [100.0 - i for i in range(1, 12)]
        t, _idx, share = classify(100.0, closes)
        assert t == "intraday"
        assert share < GAP_MIXED

    def test_mixed_gap_and_drift(self):
        # Half the move overnight, half during the session.
        closes = [105.0] + [105.0 + i for i in range(1, 6)]
        t, _, share = classify(100.0, closes)
        assert t == "mixed"
        assert GAP_MIXED < share <= GAP_DOMINANT

    def test_onset_is_the_trough_before_a_rally(self):
        # Down for four bars, then a sustained rally: onset must be the trough,
        # NOT the open. This is the TSLA 2026-05-11 shape verified on real data.
        closes = [99.0, 98.0, 97.0, 96.0, 99.0, 102.0, 105.0, 108.0]
        t, idx, _ = classify(100.0, closes)
        assert t == "intraday"
        assert idx == 3, "onset should be the last down bar before the run"

    def test_onset_is_the_peak_before_a_selloff(self):
        closes = [101.0, 102.0, 103.0, 100.0, 97.0, 94.0, 91.0]
        t, idx, _ = classify(100.0, closes)
        assert t == "intraday"
        assert idx == 2, "onset should be the peak before the decline"

    def test_flat_session_is_unknown(self):
        assert classify(100.0, [100.0] * 6)[0] == "unknown"


class TestTimingConsequence:
    def test_late_article_is_post_move_for_an_early_onset(self):
        """The whole point: if the move happened at the open, a 14:00 article is
        commentary, not a catalyst. Keying off the closing bar would file it as
        a candidate."""
        from datetime import UTC, datetime

        from swing.analysis.windows import windows_for

        onset = datetime(2026, 8, 27, 13, 35, tzinfo=UTC)      # 09:35 ET
        prev_close = datetime(2026, 8, 26, 20, 0, tzinfo=UTC)
        article = datetime(2026, 8, 27, 18, 0, tzinfo=UTC)     # 14:00 ET
        close_bar = datetime(2026, 8, 27, 20, 0, tzinfo=UTC)

        pre, post = windows_for("intraday", onset, prev_close)
        assert not pre.contains(article), "post-move article must not be a candidate"
        assert post.contains(article)

        # Keying off the closing bar instead would wrongly admit it.
        wrong_pre, _ = windows_for("intraday", close_bar, prev_close)
        assert wrong_pre.contains(article), "demonstrates the bug this guards against"


@pytest.mark.parametrize("swing_type", ["gap", "mixed"])
def test_gap_and_mixed_onset_at_session_open(swing_type):
    # By definition: the move began before the session, so onset is the open.
    prev, closes = (100.0, [110.0] * 8) if swing_type == "gap" else (100.0, [105.0, 106.0, 107.0, 108.0, 109.0, 110.0])
    t, idx, _ = classify(prev, closes)
    assert t == swing_type and idx == 0
