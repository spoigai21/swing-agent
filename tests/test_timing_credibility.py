"""A date with no time is not a publication instant.

Several feeds supply only a date, which becomes 00:00 UTC and therefore precedes
every intraday onset. A story published at 2pm then counts as evidence for a
9:30am move — the exact failure this system is built to prevent. An audit on
2026-09-22 found 56 such clusters, 28 of them treated as pre-move on that basis.
"""
from __future__ import annotations

from datetime import UTC, datetime

from swing.analysis.retrieval import time_is_unknown, timing_is_credible

ONSET = datetime(2026, 9, 18, 13, 30, tzinfo=UTC)          # 9:30 ET


def art(ts):
    return {"published_at": ts, "source": "cnbc", "headline": "h", "tickers": ["QCOM"]}


class TestDetectingADateOnlyTimestamp:
    def test_exact_midnight_is_treated_as_unknown(self):
        assert time_is_unknown(art(datetime(2026, 9, 18, 0, 0, 0, tzinfo=UTC)))

    def test_any_real_time_is_not(self):
        for ts in (datetime(2026, 9, 18, 0, 0, 1, tzinfo=UTC),
                   datetime(2026, 9, 18, 0, 1, tzinfo=UTC),
                   datetime(2026, 9, 18, 14, 3, tzinfo=UTC)):
            assert not time_is_unknown(art(ts))

    def test_a_missing_timestamp_is_not_claimed_as_unknown_time(self):
        assert time_is_unknown({"published_at": None}) is False


class TestWhenAnUnknownTimeMayStillBeUsed:
    def test_an_earlier_day_is_fine_whatever_the_hour(self):
        """Two days before the move, midnight or midday, it still precedes it."""
        assert timing_is_credible(art(datetime(2026, 9, 16, 0, 0, tzinfo=UTC)), ONSET)

    def test_the_onset_day_is_not(self):
        """Here the hour is the entire question, and we do not have it."""
        assert not timing_is_credible(art(datetime(2026, 9, 18, 0, 0, tzinfo=UTC)), ONSET)

    def test_a_real_timestamp_on_the_onset_day_is_unaffected(self):
        assert timing_is_credible(art(datetime(2026, 9, 18, 11, 4, tzinfo=UTC)), ONSET)

    def test_dropping_costs_a_citation_keeping_invents_one(self):
        """The asymmetry that decides the rule: a dropped article can only lose
        evidence we might have cited; a kept one can manufacture evidence that
        did not exist before the move."""
        same_day_unknown = art(datetime(2026, 9, 18, 0, 0, tzinfo=UTC))
        assert not timing_is_credible(same_day_unknown, ONSET)


class TestItAppliesToPreMoveOnly:
    def test_post_move_evidence_is_not_filtered(self):
        """Post-move clusters are context, shown as such; the timing claim being
        protected is only "this was published BEFORE the move"."""
        import inspect

        from swing.analysis import retrieval

        src = inspect.getsource(retrieval.build_for_swing)
        assert 'timing != "pre_move" or timing_is_credible' in src
