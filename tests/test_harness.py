"""Evaluation metric correctness.

A metric that is subtly wrong produces a confident number about the wrong thing,
which is worse than no metric. Both bugs below shipped and were caught by
reading the output, so they are pinned here.
"""
from __future__ import annotations

import pytest

from swing.eval import harness


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *a, **k):
        return self

    def fetchall(self):
        return self._rows


def _patch(monkeypatch, rows):
    import contextlib

    @contextlib.contextmanager
    def _c(*a, **k):
        yield _FakeConn(rows)

    monkeypatch.setattr(harness, "connect", _c)


def _attr(cited, shown):
    return {
        "id": 1, "swing_id": 1, "shown_cluster_ids": shown,
        "payload": {"candidates": [
            {"evidence": [{"cluster_id": c} for c in cited]}]},
    }


class TestCitationValidity:
    def test_validates_against_what_was_shown_not_the_swings_own_clusters(self, monkeypatch):
        """The bug: a placebo run is shown the DONOR's clusters, so checking
        against the test swing's own set scored every valid citation invalid and
        the metric read 0.000."""
        _patch(monkeypatch, [_attr(cited=[957, 958], shown=[957, 958, 959])])
        m = harness.citation_validity()
        assert m.value == 1.0 and m.passing is True

    def test_invented_id_is_caught(self, monkeypatch):
        _patch(monkeypatch, [_attr(cited=[957, 12345], shown=[957, 958])])
        m = harness.citation_validity()
        assert m.value == 0.5 and m.passing is False

    def test_rows_without_shown_ids_are_skipped_not_failed(self, monkeypatch):
        # Pre-date the column: unjudgeable, and must not be scored as invalid.
        _patch(monkeypatch, [_attr(cited=[1], shown=None)])
        m = harness.citation_validity()
        assert m.value is None and m.n == 0

    def test_perfect_run_passes_only_at_exactly_one(self, monkeypatch):
        _patch(monkeypatch, [_attr(cited=[1, 2, 3], shown=[1, 2, 3])])
        assert harness.citation_validity().passing is True


class TestConfabulationRate:
    def test_counts_non_abstentions(self, monkeypatch):
        _patch(monkeypatch, [{"verdict": "unexplained"}] * 9 + [{"verdict": "explained"}])
        m = harness.confabulation_rate()
        assert m.value == pytest.approx(0.1) and m.passing is False   # target < 0.10

    def test_all_abstained_is_zero(self, monkeypatch):
        _patch(monkeypatch, [{"verdict": "unexplained"}] * 5)
        m = harness.confabulation_rate()
        assert m.value == 0.0 and m.passing is True

    def test_no_rows_reports_na_not_zero(self, monkeypatch):
        # Reporting 0.0 with no data would read as a pass.
        _patch(monkeypatch, [])
        m = harness.confabulation_rate()
        assert m.value is None and m.passing is None

    def test_query_is_version_filtered(self):
        """Pooling across model/prompt/config/eval-design versions is exactly
        what the version stamp exists to prevent."""
        import inspect

        src = inspect.getsource(harness.confabulation_rate)
        for field in ("model_id", "prompt_version", "config_hash", "eval_hash"):
            assert field in src, f"confabulation_rate must filter on {field}"


class TestRecallIsBlindOnly:
    def test_recall_query_restricts_to_blind_annotations(self):
        """Assisted labels are PICKED from the retrieved list, so including them
        makes recall 1.0 by construction and the metric measures nothing."""
        import inspect

        assert "a.blind" in inspect.getsource(harness.recall_at_k)
