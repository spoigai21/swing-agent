"""Coverage could not respond to new data.

`catalyst_coverage` counts annotations whose `true_article_ids` is non-empty,
written when a human annotated. The 12-month GDELT backfill added 2,742 articles
and coverage stayed at 0.709 to three decimals — not because the backfill failed
but because the metric reads a field from the past.

This re-checks those labels against today's corpus. The danger is the obvious
shortcut: auto-linking whatever ranks highest would let retrieval pick its own
ground truth, and coverage would rise by construction while meaning less.
"""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from swing.eval import recheck as R


def article(aid, vec, headline="h", source="cnbc", tier=3):
    return {"id": aid, "embedding": vec, "published_at": datetime(2026, 1, 1, tzinfo=UTC),
            "source": source, "source_tier": tier, "headline": headline}


@pytest.fixture
def fake_embed(monkeypatch):
    def embed(texts):
        return [[1.0, 0.0]]
    monkeypatch.setattr("swing.ingest.normalize._embed_texts", embed)


class TestMatching:
    def test_the_closest_article_ranks_first(self, fake_embed):
        arts = [article(1, np.array([0.0, 1.0])), article(2, np.array([1.0, 0.1]))]
        got = R.match("gta vi delay", arts, min_similarity=0.0)
        assert [c.article_id for c in got] == [2, 1]

    def test_weak_matches_are_not_shown_at_all(self, fake_embed):
        """A human reading a list of 0.1 similarities learns nothing."""
        arts = [article(1, np.array([0.0, 1.0]))]
        assert R.match("x", arts, min_similarity=0.35) == []

    def test_a_zero_vector_is_skipped_not_divided_by(self, fake_embed):
        arts = [article(1, np.array([0.0, 0.0])), article(2, np.array([1.0, 0.0]))]
        assert [c.article_id for c in R.match("x", arts, min_similarity=0.0)] == [2]

    def test_no_articles_and_no_text_are_both_empty(self, fake_embed):
        assert R.match("x", []) == []
        assert R.match("   ", [article(1, np.array([1.0, 0.0]))]) == []

    def test_only_the_top_n_reach_the_reviewer(self, fake_embed):
        arts = [article(i, np.array([1.0, 0.0])) for i in range(20)]
        assert len(R.match("x", arts, top_n=3, min_similarity=0.0)) == 3


class TestItNeverLinksByItself:
    def test_scanning_writes_nothing(self):
        import inspect

        for fn in (R.scan, R.match, R.report, R._window_articles):
            src = inspect.getsource(fn)
            assert "UPDATE" not in src and "INSERT" not in src

    def test_only_link_writes_and_it_takes_explicit_ids(self):
        import inspect

        sig = inspect.signature(R.link)
        assert list(sig.parameters) == ["swing_id", "article_ids"]
        assert "UPDATE" in inspect.getsource(R.link)

    def test_the_report_tells_the_reader_to_check_first(self, monkeypatch):
        monkeypatch.setattr(R, "scan", lambda *a, **k: [
            R.Uncovered(1, "NVDA", "2026-01-01", "a catalyst",
                        [R.Candidate(9, 0.9, datetime(2026, 1, 1, tzinfo=UTC),
                                     "cnbc", 3, "headline")])])
        out = R.report()
        assert "Read the article before linking" in out
        assert "held-out answer key" in out


class TestLinkSafety:
    def test_it_refuses_unknown_article_ids(self, monkeypatch):
        class Conn:
            def execute(self, q, args=None):
                return self
            def fetchall(self):
                return []          # nothing matched
            def fetchone(self):
                return None
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        monkeypatch.setattr("swing.store.session.connect", lambda *a, **k: Conn())
        assert "no such article id" in R.link(1, [999])

    def test_it_never_overwrites_an_existing_label(self):
        """The UPDATE is guarded on the label still being empty, so a re-run
        cannot quietly replace ground truth someone already established."""
        import inspect

        src = inspect.getsource(R.link)
        assert "cardinality(true_article_ids), 0) = 0" in src

    def test_it_records_that_the_label_came_from_a_recheck(self):
        import inspect

        assert "recheck:" in inspect.getsource(R.link)


class TestReporting:
    def test_a_swing_with_no_window_says_so_rather_than_looking_covered(self):
        u = R.Uncovered(1, "TSLA", "2026-05-11", "catalyst text")
        u.note = "no admissible articles in the pre-move window at all"
        assert "no admissible articles" in u.block()

    def test_a_swing_with_no_match_is_still_listed_as_uncovered(self):
        assert "still uncovered" in R.Uncovered(1, "NVDA", "2026-01-01", "x").block()

    def test_nothing_uncovered_is_good_news_not_an_error(self, monkeypatch):
        monkeypatch.setattr(R, "scan", lambda *a, **k: [])
        assert "No uncovered labels" in R.report()
