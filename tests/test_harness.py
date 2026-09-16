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

    def test_recall_passes_at_the_gate_2_mark(self, monkeypatch):
        # Gate 2's overall mark is the re-baselined 0.68 (CODEBASE-PLAN 16.13),
        # the product of the coverage and covered-recall targets.
        _patch(monkeypatch, [{"swing_id": i, "rank": 1} for i in range(4)]
               + [{"swing_id": 9, "rank": None}])
        m = harness.recall_at_k()
        assert m.value == pytest.approx(0.8) and m.passing is True

    def test_overall_recall_fails_below_the_derived_mark(self, monkeypatch):
        # 0.60 is above the 0.545 we actually score and still must FAIL, so the
        # gate cannot be quietly retargeted to whatever the system happens to hit.
        _patch(monkeypatch, [{"swing_id": i, "rank": 1} for i in range(3)]
               + [{"swing_id": 8, "rank": None}, {"swing_id": 9, "rank": None}])
        m = harness.recall_at_k()
        assert m.value == pytest.approx(0.6) and m.passing is False


class TestGate2SplitsDataFromSystem:
    """One number hid two problems: 24 of 25 misses were missing evidence and
    exactly one was a ranking failure. Coverage and ranking must not mask each
    other, so each is measured on its own denominator."""

    def test_coverage_counts_an_empty_article_list_as_uncovered(self, monkeypatch):
        # An annotator names the catalyst in free text even when no article for
        # it was ever collected; that is a data gap, not a retrieval gap.
        _patch(monkeypatch, [{"n_articles": 2}, {"n_articles": 0},
                             {"n_articles": 1}, {"n_articles": 0}])
        m = harness.catalyst_coverage()
        assert m.value == pytest.approx(0.5) and m.n == 4 and m.passing is False

    def test_covered_recall_excludes_uncollected_catalysts(self):
        # The covered metric must filter on true_article_ids, or the archive's
        # gaps land in the retrieval engine's score.
        import inspect

        src = inspect.getsource(harness.recall_at_k_covered)
        assert "cardinality(a.true_article_ids)" in src and "a.blind" in src

    def test_covered_recall_holds_ranking_to_the_higher_bar(self, monkeypatch):
        # 0.80 passes the overall 0.68 mark but must FAIL the covered 0.85 one:
        # with the evidence present, failing to rank it is a real defect.
        _patch(monkeypatch, [{"swing_id": i, "rank": 1} for i in range(4)]
               + [{"swing_id": 9, "rank": None}])
        m = harness.recall_at_k_covered()
        assert m.value == pytest.approx(0.8) and m.passing is False

    def test_the_overall_target_is_the_product_of_the_two(self, monkeypatch):
        # Derived, not chosen: 0.80 coverage x 0.85 covered-recall = 0.68.
        _patch(monkeypatch, [{"n_articles": 1}])
        coverage = float(harness.catalyst_coverage().target.split()[-1])
        _patch(monkeypatch, [{"swing_id": 1, "rank": 1}])
        covered = float(harness.recall_at_k_covered().target.split()[-1])
        overall = float(harness.recall_at_k().target.split()[-1])
        assert overall == pytest.approx(coverage * covered, abs=0.005)

    def test_report_shows_both_halves(self, monkeypatch):
        _patch(monkeypatch, [])
        names = [m.name for m in harness.report()]
        assert "catalyst coverage" in names and "recall@10 (covered)" in names


def _payload(*candidates):
    return {"candidates": [{"evidence": [{"cluster_id": c} for c in cited]}
                           for cited in candidates]}


class TestLabelsSurviveRebuild:
    """Clusters are rebuilt whenever ranking is retuned. A label pinned to a
    cluster id blocked the rebuild (foreign key) and froze recall at the ranking
    that existed when the swing was annotated."""

    def test_recall_resolves_labels_through_articles(self):
        import inspect

        src = inspect.getsource(harness.recall_at_k)
        assert "true_article_ids" in src and "pre_move" in src

    def test_rebuild_upserts_instead_of_delete_and_reinsert(self):
        import inspect

        from swing.analysis import retrieval

        src = inspect.getsource(retrieval._persist) + retrieval.UPSERT_CLUSTER
        assert "ON CONFLICT (swing_id, timing, canonical_article)" in src
        assert 'DELETE FROM clusters WHERE swing_id=%s"' not in src

    def test_accuracy_hit_on_cluster_id(self):
        assert harness.top_candidate_hit(_payload([7]), 7, None, {})

    def test_accuracy_hit_after_cluster_was_re_ided(self):
        # The labelled cluster was replaced (true_cluster_id SET NULL); the
        # attribution cites the new cluster, which holds the same article.
        assert harness.top_candidate_hit(_payload([42]), None, [1001, 1002],
                                         {42: {1002, 1003}})

    def test_accuracy_miss_on_unrelated_story(self):
        assert not harness.top_candidate_hit(_payload([42]), None, [1001], {42: {9}})

    def test_only_the_top_candidate_counts(self):
        assert not harness.top_candidate_hit(_payload([1], [2]), 2, None, {})
