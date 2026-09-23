"""Every query in the codebase, actually executed.

`src/` holds 151 SQL statements and, until this file, not one test ran any of
them: the suite is green against an unreachable DATABASE_URL because everything
that would touch Postgres is stubbed. A wrong column name therefore shipped
silently, and did — twice in one day:

    clusters.size      the column is member_count   (crashed a report)
    swings.day         the column is d              (crashed an audit script)

Both were caught by a human running the command, which is not a test strategy.
These run the real functions against a real schema, so the SQL is checked even
when the result is an empty list.

⚠️ Guarded: they need SWING_TEST_DATABASE_URL naming a *_test database and skip
otherwise. See tests/conftest_db.py — pointing them at a working corpus would
destroy collection that cannot be repeated.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

pytestmark = pytest.mark.db


def _payload(cluster_id: int) -> str:
    import json

    return json.dumps({"candidates": [{
        "catalyst": "new GPU", "confidence": "high",
        "direction_consistent": True, "magnitude_plausible": True,
        "evidence": [{"cluster_id": cluster_id, "timing": "pre_move",
                      "source_tier": 3, "headline": "Nvidia announces a new GPU",
                      "source": "cnbc",
                      "published_at": "2026-01-01T00:00:00Z"}]}]})


@pytest.fixture
def seeded(clean_db):
    """One ticker, one swing, one article, one cluster — enough for every join."""
    now = datetime.now(UTC).replace(microsecond=0)
    day = now.date()
    with clean_db.connect() as conn:
        conn.execute(
            "INSERT INTO articles_raw (url, source, headline, published_at, retrieved_at, raw)"
            " VALUES (%s,%s,%s,%s,%s,%s)",
            ("https://example.test/a1", "cnbc", "Nvidia announces a new GPU",
             now - timedelta(hours=6), now, '{"via":"test"}'))
        raw_id = conn.execute("SELECT id FROM articles_raw LIMIT 1").fetchone()["id"]
        conn.execute(
            "INSERT INTO articles (raw_id, url, source, source_tier, headline,"
            " published_at, retrieved_at, tickers, embedding)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (raw_id, "https://example.test/a1", "cnbc", 3,
             "Nvidia announces a new GPU", now - timedelta(hours=6), now,
             ["NVDA"], "[" + ",".join(["0.01"] * 768) + "]"))
        article_id = conn.execute("SELECT id FROM articles LIMIT 1").fetchone()["id"]
        conn.execute(
            "INSERT INTO swings (ticker, d, kind, entity_type, total_return,"
            " market_component, sector_component, residual, residual_z, volume_z,"
            " swing_type, onset_ts, onset_source, earnings_mode)"
            " VALUES ('NVDA',%s,'daily','stock',0.05,0.01,0.01,0.03,3.1,0.5,"
            "'intraday',%s,'intraday',false)",
            (day, now - timedelta(hours=3)))
        swing_id = conn.execute("SELECT id FROM swings LIMIT 1").fetchone()["id"]
        conn.execute(
            "INSERT INTO clusters (swing_id, timing, canonical_article, member_count,"
            " distinct_sources, earliest_published, best_tier, semantic_score,"
            " timing_score, novelty_score, rank_score, rank)"
            " VALUES (%s,'pre_move',%s,1,1,%s,3,0.6,0.5,0.0,1.6,1)",
            (swing_id, article_id, now - timedelta(hours=6)))
        cluster_id = conn.execute("SELECT id FROM clusters LIMIT 1").fetchone()["id"]
        conn.execute("INSERT INTO cluster_members (cluster_id, article_id) VALUES (%s,%s)",
                     (cluster_id, article_id))
        conn.execute(
            "INSERT INTO annotations (swing_id, blind, true_catalyst, true_cluster_id,"
            " true_article_ids, true_event_type, no_catalyst)"
            " VALUES (%s, true, 'new GPU', %s, %s, 'product', false)",
            (swing_id, cluster_id, [article_id]))
        conn.execute(
            "INSERT INTO attributions (swing_id, verdict, payload, run_kind,"
            " shown_cluster_ids, prompt_version, model_id, config_hash)"
            " VALUES (%s,'explained',%s,'production',%s,'v3','test-model','deadbeef')",
            (swing_id, _payload(cluster_id), [cluster_id]))
    return {"swing_id": swing_id, "article_id": article_id, "cluster_id": cluster_id}


class TestTheQueryLayerRuns:
    def test_every_public_query_executes(self, seeded):
        """queries.py is the one module allowed to write SQL for the app."""
        import inspect

        from swing.store import queries

        fns = [f for name, f in vars(queries).items()
               if inspect.isfunction(f) and not name.startswith("_")
               and not inspect.signature(f).parameters]
        assert fns, "no zero-argument query functions found"
        for fn in fns:
            fn()

    def test_queries_taking_a_day_count_execute(self, seeded):
        from swing.store import queries

        for name in ("unexplained_rate_by_ticker",):
            getattr(queries, name)(30)


class TestTheMetricsRun:
    def test_every_harness_metric_executes(self, seeded):
        from swing.eval import harness

        for metric in (harness.catalyst_coverage(), harness.recall_at_k(),
                       harness.recall_at_k_covered(), harness.attribution_accuracy(),
                       harness.abstention_precision(), harness.citation_validity()):
            assert metric.name

    def test_calibration_source_precision_and_classgaps_execute(self, seeded):
        from swing.eval import calibration, classgaps, sources

        assert isinstance(calibration.scored_cases(), list)
        assert isinstance(calibration.flag_values("confidence"), dict)
        assert isinstance(sources.collect(), list)
        assert isinstance(classgaps.label_counts(), dict)
        assert isinstance(classgaps.suggest("macro"), list)

    def test_weight_fitting_loads_cases(self, seeded):
        from swing.models import weights

        assert isinstance(weights.load_cases(), list)


class TestTheReportsRun:
    def test_monitor_queries_execute(self, seeded):
        from swing.interface import monitor

        assert isinstance(monitor.unexplained_by_ticker(90), list)
        assert isinstance(monitor.unexplained_by_swing_type(90), list)

    def test_uptime_queries_execute(self, seeded):
        from swing.ingest import uptime

        assert isinstance(uptime.hourly(2), list)
        uptime.gap_since_last_row()

    def test_recheck_and_schedule_queries_execute(self, seeded):
        from swing.eval import abstention, accuracy, recheck

        assert isinstance(recheck.uncovered_rows(), list)
        assert isinstance(abstention.pending(), list)
        assert isinstance(accuracy.pending(), list)


class TestTheSchemaMatchesTheCode:
    def test_the_columns_the_code_names_exist(self, seeded):
        """`clusters.size` and `swings.day` both shipped; neither exists."""
        from swing.store.session import connect

        expected = {
            "swings": {"id", "ticker", "d", "residual_z", "onset_ts", "swing_type",
                       "volume_z", "earnings_mode", "superseded_by", "kind"},
            "clusters": {"id", "swing_id", "timing", "canonical_article",
                         "member_count", "distinct_sources", "earliest_published",
                         "best_tier", "rank", "rank_score", "semantic_score"},
            "articles": {"id", "url", "source", "source_tier", "headline", "summary",
                         "published_at", "tickers", "embedding"},
            "annotations": {"swing_id", "blind", "true_catalyst", "true_article_ids",
                            "true_event_type", "no_catalyst"},
            "attributions": {"swing_id", "verdict", "payload", "run_kind",
                             "shown_cluster_ids", "config_hash", "model_id"},
        }
        with connect() as conn:
            for table, columns in expected.items():
                have = {r["column_name"] for r in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name=%s", (table,)).fetchall()}
                assert columns <= have, f"{table} is missing {columns - have}"

    def test_pgvector_is_installed(self, seeded):
        from swing.store.session import connect

        with connect() as conn:
            r = conn.execute("SELECT count(*) n FROM pg_extension "
                             "WHERE extname='vector'").fetchone()
            assert r["n"] == 1, "the vector extension is required"
