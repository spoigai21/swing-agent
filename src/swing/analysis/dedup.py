"""Two-tier deduplication. agent-plan.md 2.2.

Without this the agent reports "eight sources corroborate this" when it is one
story reprinted eight times. Design Rule 4: a syndicated wire story is one
source, not thirty.

Two tiers, because agglomerative clustering is not incremental while
`articles.cluster_id` as specified implies it is (CODEBASE-PLAN G4):

  * **global, incremental** — MinHash LSH at normalize time -> `dup_group_id`.
    Stable, cheap, catches verbatim reprints. Runs once per article, forever.
  * **window-scoped, at retrieval** — agglomerative clustering over one swing's
    pre/post windows -> rows in `clusters` + `cluster_members`. Persisted, so a
    citation stays resolvable even after thresholds are retuned.

NOTE for this stack: there is no tier-2 wire copy to collapse (CODEBASE-PLAN 12),
so the semantic pass mostly merges tier-3 rewrites of the same story. Still
worth doing, just less load-bearing than the plan assumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from swing.common import logging as log
from swing.common.vectors import as_array
from swing.ingest.config import thresholds

logger = log.get("analysis.dedup")


@dataclass(slots=True)
class ClusterView:
    """What the agent reasons over. It never sees raw articles."""
    timing: str
    article_ids: list[int]
    canonical_article: int
    headline: str
    source: str
    member_count: int
    distinct_sources: int
    earliest_published: datetime
    best_tier: int
    semantic_score: float = 0.0
    timing_score: float = 0.0
    novelty_score: float = 0.0
    rank_score: float = 0.0
    rank: int | None = None
    members: list[dict] = field(default_factory=list)


def _cosine_matrix(vecs: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vecs, axis=1, keepdims=True)
    n[n == 0] = 1.0
    unit = vecs / n
    return unit @ unit.T


def cluster_window(articles: list[dict], timing: str,
                   cosine_threshold: float | None = None) -> list[ClusterView]:
    """Single-link agglomerative clustering on embeddings within one window.

    Single-link (connect anything above threshold, take transitive closure) is
    the right choice for near-duplicate detection: a story rewritten twice
    should land in one cluster even if the two rewrites are less similar to each
    other than to the original.
    """
    if not articles:
        return []
    thr = cosine_threshold if cosine_threshold is not None else float(
        thresholds()["dedup"]["semantic_cosine"])

    vecs = np.vstack([as_array(a["embedding"]) for a in articles])
    sim = _cosine_matrix(vecs)

    # Union-find over the above-threshold graph.
    parent = list(range(len(articles)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(articles)):
        for j in range(i + 1, len(articles)):
            # An exact MinHash duplicate is merged regardless of cosine.
            same_dup = (articles[i].get("dup_group_id") is not None
                        and articles[i]["dup_group_id"] == articles[j].get("dup_group_id"))
            if same_dup or sim[i, j] >= thr:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(len(articles)):
        groups.setdefault(find(idx), []).append(idx)

    out = []
    for members in groups.values():
        rows = [articles[i] for i in members]
        # Canonical = earliest published, tie-broken by best (lowest) tier.
        canon = min(rows, key=lambda r: (r["published_at"], r["source_tier"]))
        out.append(ClusterView(
            timing=timing,
            article_ids=[r["id"] for r in rows],
            canonical_article=canon["id"],
            headline=canon["headline"],
            source=canon["source"],
            member_count=len(rows),
            # THE corroboration count: distinct publishers, not article count.
            distinct_sources=len({r["source"] for r in rows}),
            earliest_published=min(r["published_at"] for r in rows),
            best_tier=min(r["source_tier"] for r in rows),
            members=rows,
        ))
    return out
