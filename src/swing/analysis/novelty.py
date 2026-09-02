"""Novelty scoring. Heuristic first — no training required. agent-plan.md 5.1.

Prices react to SURPRISE, not information. An article restating three weeks of
existing coverage is not a catalyst even when it is highly relevant.

    novelty = 1 - max_cosine_similarity(cluster, trailing_30d_corpus)

The MLP in models/train_novelty.py must beat this heuristic on recall@10 or the
heuristic ships. Until Phase 5 proves otherwise, ranking uses w_novelty = 0.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np

from swing.analysis.dedup import ClusterView
from swing.common.vectors import as_array
from swing.store.session import connect


def trailing_corpus(tickers: list[str], before, days: int = 30,
                    exclude_after=None) -> np.ndarray:
    """Embeddings for this name's PRIOR coverage.

    ⚠️ `exclude_after` must be the start of the retrieval window. Without it the
    corpus contains the very articles being scored, every cluster matches itself
    at cosine 1.0, and novelty is uniformly 0 — which silently disables the
    signal while looking like it works.
    """
    cutoff = exclude_after or before
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT embedding FROM articles
            WHERE tickers && %s AND published_at < %s AND published_at >= %s
              AND embedding IS NOT NULL
            """,
            (tickers, cutoff, before - timedelta(days=days)),
        ).fetchall()
    if not rows:
        return np.zeros((0, 768))
    return np.array([as_array(r["embedding"]) for r in rows], dtype=float)


def score(cluster: ClusterView, corpus: np.ndarray) -> float:
    """1.0 = nothing like it in the trailing corpus. 0.0 = pure restatement."""
    if corpus.shape[0] == 0:
        return 1.0
    vec = as_array(cluster.members[0]["embedding"])
    vn = np.linalg.norm(vec)
    if vn == 0:
        return 1.0
    cn = np.linalg.norm(corpus, axis=1)
    cn[cn == 0] = 1.0
    sims = (corpus / cn[:, None]) @ (vec / vn)
    return float(max(0.0, 1.0 - float(np.max(sims))))
