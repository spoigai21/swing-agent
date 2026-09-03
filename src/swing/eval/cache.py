"""Persisted cluster payloads so an eval re-run hits only the LLM.

agent-plan.md 4.3: a 200-case sweep is re-run after every prompt change. Without
this, each sweep re-queries and re-ranks; with it, only the model call repeats.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from swing.ingest.config import thresholds
from swing.store.session import connect


@lru_cache(maxsize=2048)
def _fetch(swing_id: int) -> tuple:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.timing, c.best_tier, c.distinct_sources, c.member_count,
                   c.earliest_published, c.rank, a.headline, a.summary, a.source
            FROM clusters c JOIN articles a ON a.id = c.canonical_article
            WHERE c.swing_id=%s ORDER BY c.timing, c.rank
            """,
            (swing_id,),
        ).fetchall()
    return tuple(dict(r) for r in rows)


def cached_clusters(swing_id: int) -> dict[str, list[dict[str, Any]]]:
    """The cluster payload for a swing, split by timing and capped at top_k."""
    top_k = int(thresholds()["retrieval"]["max_clusters_to_llm"])
    rows = _fetch(swing_id)
    return {
        "pre_move": [r for r in rows if r["timing"] == "pre_move"][:top_k],
        "post_move": [r for r in rows if r["timing"] == "post_move"][:top_k],
    }


def clear() -> None:
    _fetch.cache_clear()
