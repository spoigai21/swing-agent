"""Assemble a swing's candidate clusters end to end.

  swing -> pre/post windows -> articles -> clusters -> ranked -> persisted

The agent reasons over CLUSTERS, never raw articles (agent-plan.md 2.2), and the
pre/post split is never merged (2.1) — that split is the backbone of the
system's honesty.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta

import numpy as np

from swing.analysis.dedup import ClusterView, cluster_window
from swing.analysis.novelty import score as novelty_score
from swing.analysis.novelty import trailing_corpus
from swing.analysis.rank import score_clusters
from swing.analysis.windows import windows_for
from swing.common import logging as log
from swing.common.timeutil import UTC
from swing.ingest.config import stocks, thresholds
from swing.store.session import connect

logger = log.get("analysis.retrieval")


def _articles_in(tickers: list[str], start: datetime, end: datetime,
                 max_tier: int = 3) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, url, source, source_tier, headline, summary, published_at,
                   tickers, event_hint, embedding, dup_group_id
            FROM articles
            WHERE tickers && %s AND published_at >= %s AND published_at < %s
              AND source_tier <= %s AND embedding IS NOT NULL
            ORDER BY published_at
            """,
            (tickers, start, end, max_tier),
        ).fetchall()
    return [dict(r) for r in rows]


def query_text(swing: dict) -> str:
    """The query a cluster's relevance is measured against.

    Articles are already ticker-filtered, so relevance must separate
    market-moving material from routine company PR. Anchoring the query on the
    stock move and its financial vocabulary does that: without it every cluster
    scores a flat 0.5, ranking degenerates to timing alone, and a product press
    release outranks the earnings 8-K purely for being closer to the open.
    """
    from swing.ingest.config import sectors, stocks

    if swing.get("entity_type") == "sector":
        name = (sectors().get(swing["ticker"], {}) or {}).get("name", swing["ticker"])
    else:
        name = (stocks().get(swing["ticker"], {}) or {}).get("name", swing["ticker"])
    direction = "rose" if float(swing["total_return"]) >= 0 else "fell"
    pct = abs(float(swing["total_return"])) * 100
    return (f"{swing['ticker']} {name} stock {direction} {pct:.1f}% — "
            "earnings, guidance, revenue, analyst rating, regulatory or "
            "acquisition news moving the share price")


def _query_vec(swing: dict):
    from swing.ingest.normalize import _embed_texts

    try:
        return np.array(_embed_texts([query_text(swing)])[0], dtype=float)
    except Exception:
        logger.exception("query embedding failed for swing %s", swing["id"])
        return None


def _swing_row(swing_id: int) -> dict | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM swings WHERE id=%s", (swing_id,)).fetchone()


def _prev_close_ts(ticker: str, d) -> datetime:
    with connect() as conn:
        r = conn.execute(
            "SELECT ts FROM bars WHERE ticker=%s AND ts::date < %s ORDER BY ts DESC LIMIT 1",
            (ticker, d)).fetchone()
    # Bars are stamped at midnight UTC; the session actually closed at 20:00 UTC.
    base = r["ts"].date() if r else (d - timedelta(days=1))
    return datetime.combine(base, time(20, 0), tzinfo=UTC)


def own_tickers(swing: dict) -> list[str]:
    """The tags that make an article THIS entity's own news.

    A sector ETF is never itself tagged on an article — nobody writes "XLC" in a
    headline — so searching for its own symbol returns nothing and every sector
    swing abstains for lack of evidence. agent-plan.md 0.1b: retrieval for a
    sector entity searches INDUSTRY-level news instead, approximated by the
    constituents we track, which is exactly the set that defines the sector
    residual.
    """
    if swing.get("entity_type") != "sector":
        return [swing["ticker"]]
    etf = swing["ticker"]
    members = [t for t, m in stocks().items() if m.get("sector_etf") == etf]
    return members or [etf]


def retrieval_tickers(swing: dict) -> list[str]:
    """Own tags first, then related companies for a stock.

    News about another company (a rival's results, a customer's chip plans) is
    often the cause of a stock's move and never names the stock, so a stock
    also searches its related companies (config/watchlist.yaml `related`).
    Sectors do not: their constituents already are the industry.
    """
    from swing.ingest.config import related

    own = own_tickers(swing)
    if swing.get("entity_type") == "sector":
        return own
    return own + [t for t in related(swing["ticker"]) if t not in own]


def admissible(article: dict, own: list[str]) -> bool:
    """Drop a related company's periodic reports from a stock's evidence.

    Its 8-K is the event; the 10-Q/10-K filed alongside restates the quarter. On
    NVDA 2026-04-30 Meta's and Amazon's 10-Qs ranked first and second, above
    everything about Nvidia itself.
    """
    if article.get("source") != "sec-edgar" or set(article.get("tickers") or []) & set(own):
        return True
    return not any(f" {form}" in article.get("headline", "") for form in ("10-Q", "10-K"))


def build_for_swing(swing_id: int, persist: bool = True) -> dict[str, list[ClusterView]]:
    swing = _swing_row(swing_id)
    if not swing:
        raise ValueError(f"no swing {swing_id}")
    if swing["onset_ts"] is None:
        logger.warning("swing %s has no onset_ts; skipping", swing_id)
        return {"pre_move": [], "post_move": []}

    tickers = retrieval_tickers(dict(swing))
    onset = swing["onset_ts"]
    prev_close = _prev_close_ts(swing["ticker"], swing["d"])
    session_open = datetime.combine(swing["d"], time(13, 30), tzinfo=UTC)
    drift_start = None
    if swing["kind"] == "drift" and swing["drift_window"]:
        drift_start = onset - timedelta(days=int(swing["drift_window"]))

    pre_w, post_w = windows_for(swing["swing_type"], onset, prev_close,
                                session_open, drift_start)
    max_tier = int(thresholds()["retrieval"]["exclude_tier"]) - 1

    out: dict[str, list[ClusterView]] = {}
    own = own_tickers(dict(swing))
    corpus = trailing_corpus(own, onset, exclude_after=pre_w.start)
    qvec = _query_vec(swing)
    for timing, w in (("pre_move", pre_w), ("post_move", post_w)):
        arts = [a for a in _articles_in(tickers, w.start, w.end, max_tier)
                if admissible(a, own)]
        clusters = cluster_window(arts, timing)
        if timing == "pre_move":
            nov = {c.canonical_article: novelty_score(c, corpus) for c in clusters}
        else:
            nov = None
        # Rank pre-move on the full score; post-move is context, ordered by time.
        if timing == "pre_move":
            clusters = score_clusters(clusters, onset, qvec, nov, own=set(own))
        else:
            clusters = sorted(clusters, key=lambda c: c.earliest_published)
            for i, c in enumerate(clusters, start=1):
                c.rank = i
        out[timing] = clusters

    if persist:
        _persist(swing_id, out)
    return out


UPSERT_CLUSTER = """
INSERT INTO clusters (swing_id, timing, canonical_article, member_count,
                      distinct_sources, earliest_published, best_tier,
                      semantic_score, timing_score, novelty_score, rank_score, rank)
VALUES (%(swing_id)s, %(timing)s, %(canonical_article)s, %(member_count)s,
        %(distinct_sources)s, %(earliest_published)s, %(best_tier)s,
        %(semantic_score)s, %(timing_score)s, %(novelty_score)s, %(rank_score)s, %(rank)s)
ON CONFLICT (swing_id, timing, canonical_article) DO UPDATE SET
  member_count=EXCLUDED.member_count, distinct_sources=EXCLUDED.distinct_sources,
  earliest_published=EXCLUDED.earliest_published, best_tier=EXCLUDED.best_tier,
  semantic_score=EXCLUDED.semantic_score, timing_score=EXCLUDED.timing_score,
  novelty_score=EXCLUDED.novelty_score, rank_score=EXCLUDED.rank_score,
  rank=EXCLUDED.rank
RETURNING id
"""


def _persist(swing_id: int, groups: dict[str, list[ClusterView]]) -> None:
    """Upsert keyed on (swing_id, timing, canonical_article), in one transaction.

    ⚠️ Not delete-and-reinsert. That gave every cluster a new id on each rebuild,
    and annotations.true_cluster_id references clusters(id), so the DELETE failed
    for any labelled swing: build_all() logged it and moved on, and the labelled
    swing kept its old ranking forever. Retuning weights after Gate 2 would then
    leave recall@10 frozen whatever the weights were.

    A weight retune changes only rank, so ids now survive it. A dedup-threshold
    retune can change a cluster's canonical article; that cluster is replaced,
    and the label survives through annotations.true_article_ids.
    """
    kept: list[int] = []
    with connect() as conn, conn.transaction(), conn.cursor() as cur:
        for timing, clusters in groups.items():
            for c in clusters:
                cur.execute(UPSERT_CLUSTER, {
                    "swing_id": swing_id, "timing": timing,
                    "canonical_article": c.canonical_article,
                    "member_count": c.member_count,
                    "distinct_sources": c.distinct_sources,
                    "earliest_published": c.earliest_published,
                    "best_tier": c.best_tier,
                    "semantic_score": c.semantic_score, "timing_score": c.timing_score,
                    "novelty_score": c.novelty_score, "rank_score": c.rank_score,
                    "rank": c.rank,
                })
                cid = cur.fetchone()["id"]
                kept.append(cid)
                cur.execute("DELETE FROM cluster_members WHERE cluster_id=%s", (cid,))
                cur.executemany(
                    "INSERT INTO cluster_members (cluster_id, article_id) VALUES (%s, %s) "
                    "ON CONFLICT DO NOTHING",
                    [(cid, aid) for aid in c.article_ids])
        # Clusters that no longer exist. A label pointing at one is SET NULL and
        # still resolves through annotations.true_article_ids.
        cur.execute("DELETE FROM clusters WHERE swing_id=%s AND NOT (id = ANY(%s::bigint[]))",
                    (swing_id, kept))


def build_all(limit: int | None = None, only_with_articles: bool = True) -> dict[str, int]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id FROM swings WHERE onset_ts IS NOT NULL ORDER BY d DESC"
            + (f" LIMIT {int(limit)}" if limit else "")
        ).fetchall()
    built = with_pre = 0
    for r in rows:
        try:
            g = build_for_swing(r["id"])
        except Exception:
            logger.exception("cluster build failed for swing %s", r["id"])
            continue
        built += 1
        if g["pre_move"]:
            with_pre += 1
    logger.info("built clusters for %d swings (%d have pre-move coverage)", built, with_pre)
    return {"swings": built, "with_pre_move": with_pre}
