"""Cluster ranking. agent-plan.md 2.3.

    score = w_sem  * semantic_relevance
          + w_time * timing_score      # decays before onset; 0 if after
          + w_tier * (5 - tier)
          + w_nov  * novelty           # 0.0 until Phase 5.1 beats its baseline
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from swing.analysis.dedup import ClusterView
from swing.common.vectors import as_array
from swing.ingest.config import thresholds


def centroid(cluster: ClusterView) -> np.ndarray | None:
    """Mean of the members' embeddings — what the cluster is ABOUT.

    ⚠️ This used to be `members[0]` alone. A cluster is a story told by many
    outlets, and its canonical member is whichever article happened to be
    picked; judging a 12-article cluster by one of them throws away the other
    eleven and makes relevance depend on an arbitrary choice. Averaging is the
    standard summary of a set of embeddings and costs nothing at this scale.
    """
    vecs = [as_array(m["embedding"]) for m in cluster.members
            if m.get("embedding") is not None]
    vecs = [v for v in vecs if v is not None and v.size and np.linalg.norm(v) > 0]
    if not vecs:
        return None
    return np.mean(np.stack(vecs), axis=0)


def semantic_relevance(cluster: ClusterView, query_vec: np.ndarray | None) -> float:
    """Cosine against a query embedding, rescaled from [-1,1] to [0,1].

    With no query vector this returns a neutral 0.5 rather than 0, so ranking
    falls back to timing and tier instead of collapsing to a single term.
    """
    if query_vec is None:
        return 0.5
    v = centroid(cluster)
    if v is None:
        return 0.5
    nv, nq = np.linalg.norm(v), np.linalg.norm(query_vec)
    if nv == 0 or nq == 0:
        return 0.5
    return float((float(v @ query_vec / (nv * nq)) + 1.0) / 2.0)


def timing_score(cluster: ClusterView, onset: datetime) -> float:
    """Exponential decay with distance BEFORE onset. Zero at or after onset.

    Timing is evidence (Design Rule 2). An article published after the move is
    commentary, and must never earn ranking credit.
    """
    cfg = thresholds()["ranking"]
    half = float(cfg["timing_halflife_hours"])
    delta_h = (onset - cluster.earliest_published).total_seconds() / 3600.0
    if delta_h <= 0:
        return 0.0
    return float(math.pow(0.5, delta_h / half))


def score_clusters(clusters: list[ClusterView], onset: datetime,
                   query_vec: np.ndarray | None = None,
                   novelty: dict[int, float] | None = None,
                   own: set[str] | None = None) -> list[ClusterView]:
    """Score, sort and assign ranks in place. Returns the sorted list.

    `own` is the entity's own ticker tags. A cluster none of whose articles is
    tagged with one is only about a related company, and pays
    `w_related_penalty` so the stock's own news leads at equal relevance.
    """
    cfg = thresholds()["ranking"]
    w_sem, w_time = float(cfg["w_semantic"]), float(cfg["w_timing"])
    w_tier, w_nov = float(cfg["w_tier"]), float(cfg["w_novelty"])
    w_related = float(cfg.get("w_related_penalty", 0.0))

    for c in clusters:
        c.semantic_score = semantic_relevance(c, query_vec)
        c.timing_score = timing_score(c, onset)
        if novelty is not None:
            c.novelty_score = novelty.get(c.canonical_article, 0.0)
        related_only = own is not None and not any(
            set(m.get("tickers") or []) & own for m in c.members)
        c.rank_score = (
            w_sem * c.semantic_score
            + w_time * c.timing_score
            + w_tier * (5 - c.best_tier)
            + w_nov * c.novelty_score
            - (w_related if related_only else 0.0)
        )

    ordered = sorted(clusters, key=lambda c: (-c.rank_score, c.earliest_published))
    for i, c in enumerate(ordered, start=1):
        c.rank = i
    return ordered
