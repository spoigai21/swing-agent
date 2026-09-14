"""Clustering and ranking. Corroboration counting and timing are load-bearing."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from swing.analysis.dedup import ClusterView, cluster_window
from swing.analysis.rank import score_clusters, timing_score

ONSET = datetime(2026, 8, 27, 14, 0, tzinfo=UTC)


def _art(i, vec, source="wsj", tier=3, minutes_before=60, dup=None):
    return {
        "id": i, "source": source, "source_tier": tier,
        "headline": f"headline {i}", "summary": "",
        "published_at": ONSET - timedelta(minutes=minutes_before),
        "embedding": np.array(vec, dtype=float), "dup_group_id": dup,
    }


def _v(*xs):
    a = np.zeros(8)
    for i, x in enumerate(xs):
        a[i] = x
    return a / np.linalg.norm(a)


class TestClustering:
    def test_identical_articles_merge(self):
        arts = [_art(1, _v(1, 0)), _art(2, _v(1, 0), source="cnbc")]
        cl = cluster_window(arts, "pre_move", cosine_threshold=0.9)
        assert len(cl) == 1
        assert cl[0].member_count == 2

    def test_unrelated_articles_stay_separate(self):
        arts = [_art(1, _v(1, 0)), _art(2, _v(0, 1))]
        cl = cluster_window(arts, "pre_move", cosine_threshold=0.9)
        assert len(cl) == 2

    def test_distinct_sources_is_publishers_not_articles(self):
        # The corroboration count. Eight reprints of one wire story is ONE
        # source, not eight. Design Rule 4.
        arts = [_art(1, _v(1, 0), source="wsj"),
                _art(2, _v(1, 0), source="wsj"),
                _art(3, _v(1, 0), source="cnbc")]
        cl = cluster_window(arts, "pre_move", cosine_threshold=0.9)
        assert cl[0].member_count == 3
        assert cl[0].distinct_sources == 2

    def test_minhash_duplicates_merge_regardless_of_cosine(self):
        arts = [_art(1, _v(1, 0), dup=42), _art(2, _v(0, 1), dup=42)]
        cl = cluster_window(arts, "pre_move", cosine_threshold=0.99)
        assert len(cl) == 1

    def test_canonical_is_earliest_then_best_tier(self):
        arts = [_art(1, _v(1, 0), tier=3, minutes_before=30),
                _art(2, _v(1, 0), tier=1, minutes_before=120)]
        cl = cluster_window(arts, "pre_move", cosine_threshold=0.9)
        assert cl[0].canonical_article == 2      # earliest wins
        assert cl[0].best_tier == 1              # best tier in the cluster

    def test_single_link_transitivity(self):
        # A~B and B~C but A!~C: all three belong to one story.
        a, b, c = _v(1, 0), _v(0.9, 0.44), _v(0.6, 0.8)
        cl = cluster_window([_art(1, a), _art(2, b), _art(3, c)],
                            "pre_move", cosine_threshold=0.85)
        assert len(cl) == 1 and cl[0].member_count == 3

    def test_different_companies_never_merge(self):
        # "MSFT 8-K — Item 2.02" and "AMZN 8-K — Item 2.02" embed almost
        # identically; they are still two companies' earnings.
        arts = [{**_art(1, _v(1, 0), source="cnbc"), "tickers": ["MSFT"]},
                {**_art(2, _v(1, 0), source="wsj"), "tickers": ["AMZN"]}]
        assert len(cluster_window(arts, "pre_move", cosine_threshold=0.9)) == 2

    def test_two_sec_filings_never_merge(self):
        arts = [{**_art(1, _v(1, 0), source="sec-edgar", tier=1), "tickers": ["TSLA"]},
                {**_art(2, _v(1, 0), source="sec-edgar", tier=1), "tickers": ["TSLA"]}]
        assert len(cluster_window(arts, "pre_move", cosine_threshold=0.9)) == 2

    def test_a_story_cannot_bridge_two_filings_into_one_cluster(self):
        # Filing A ~ story ~ filing B. Single-link alone would chain all three;
        # the story may join one filing, never both.
        a = {**_art(1, _v(1, 0), source="sec-edgar", tier=1), "tickers": ["TSLA"]}
        story = {**_art(2, _v(0.95, 0.31), source="cnbc"), "tickers": ["TSLA"]}
        b = {**_art(3, _v(0.81, 0.59), source="sec-edgar", tier=1), "tickers": ["TSLA"]}
        clusters = cluster_window([a, story, b], "pre_move", cosine_threshold=0.9)
        assert len(clusters) == 2
        for c in clusters:
            assert sum(m["source"] == "sec-edgar" for m in c.members) == 1

    def test_rewrites_about_the_same_company_still_merge(self):
        arts = [{**_art(1, _v(1, 0), source="sec-edgar", tier=1), "tickers": ["INTC"]},
                {**_art(2, _v(1, 0), source="cnbc"), "tickers": ["INTC", "QCOM"]}]
        assert len(cluster_window(arts, "pre_move", cosine_threshold=0.9)) == 1

    def test_empty_input(self):
        assert cluster_window([], "pre_move") == []


class TestTiming:
    def _cluster(self, published):
        return ClusterView(timing="pre_move", article_ids=[1], canonical_article=1,
                           headline="h", source="wsj", member_count=1,
                           distinct_sources=1, earliest_published=published,
                           best_tier=3, members=[_art(1, _v(1, 0))])

    def test_article_after_onset_scores_zero(self):
        # Post-move commentary must never earn ranking credit.
        c = self._cluster(ONSET + timedelta(minutes=5))
        assert timing_score(c, ONSET) == 0.0

    def test_article_at_onset_scores_zero(self):
        assert timing_score(self._cluster(ONSET), ONSET) == 0.0

    def test_closer_before_onset_scores_higher(self):
        near = timing_score(self._cluster(ONSET - timedelta(hours=1)), ONSET)
        far = timing_score(self._cluster(ONSET - timedelta(hours=48)), ONSET)
        assert 0 < far < near <= 1.0

    def test_tier_1_outranks_tier_3_all_else_equal(self):
        t1 = self._cluster(ONSET - timedelta(hours=2)); t1.best_tier = 1
        t3 = self._cluster(ONSET - timedelta(hours=2)); t3.best_tier = 3
        ranked = score_clusters([t3, t1], ONSET)
        assert ranked[0].best_tier == 1

    def test_post_move_cluster_cannot_outrank_a_pre_move_one(self):
        pre = self._cluster(ONSET - timedelta(hours=6)); pre.best_tier = 3
        post = self._cluster(ONSET + timedelta(hours=1)); post.best_tier = 3
        ranked = score_clusters([post, pre], ONSET)
        assert ranked[0] is pre


@pytest.mark.parametrize("hours,expected", [(12, 0.5), (24, 0.25)])
def test_timing_halflife_is_twelve_hours(hours, expected):
    c = ClusterView(timing="pre_move", article_ids=[1], canonical_article=1,
                    headline="h", source="wsj", member_count=1, distinct_sources=1,
                    earliest_published=ONSET - timedelta(hours=hours), best_tier=3,
                    members=[])
    assert timing_score(c, ONSET) == pytest.approx(expected, abs=0.01)


class TestSwingUniqueness:
    """UNIQUE (ticker, d, kind, drift_window) never fires for daily swings
    because drift_window is NULL and NULL != NULL in Postgres. 101 of 472 rows
    were duplicates before this was caught, and the daily batch attributed the
    same swing twice."""

    def test_schema_uses_partial_indexes_not_a_table_unique(self):
        from swing.paths import ROOT

        raw = (ROOT / "src/swing/store/schema.sql").read_text()
        # Strip comments: the explanatory note about the old constraint quotes
        # it verbatim, and a naive substring check matches the comment.
        sql = "\n".join(ln for ln in raw.splitlines() if not ln.strip().startswith("--"))
        assert "UNIQUE (ticker, d, kind, drift_window)" not in sql
        assert "swings_daily_unique" in sql and "WHERE drift_window IS NULL" in sql
        assert "swings_drift_unique" in sql

    def test_upserts_name_the_index_predicate(self):
        from swing.analysis.swings import UPSERT_DAILY, UPSERT_DRIFT

        assert "WHERE drift_window IS NULL" in UPSERT_DAILY
        assert "WHERE drift_window IS NOT NULL" in UPSERT_DRIFT
        assert "DO UPDATE" in UPSERT_DAILY and "DO UPDATE" in UPSERT_DRIFT
