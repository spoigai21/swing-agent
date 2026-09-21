"""Phase 5.3 — (swing, true catalyst) pairs for bi-encoder fine-tuning.

agent-plan.md Step 5.3: "Once you have 200+ confirmed (swing, true catalyst
cluster) pairs, fine-tune a bi-encoder with `sentence-transformers` using
contrastive loss. Positive: the true catalyst. Hard negatives: other clusters
from the same window. Baseline to beat: the off-the-shelf embedding model."

    .venv/bin/python -m swing.models.pairs

⚠️ **The stated baseline is already saturated.** On the 38 pairs that exist, the
off-the-shelf model puts the true catalyst in the top 10 every single time —
recall@10 = 1.000, worst rank 7. Nothing can beat a perfect score, so Gate 5 for
5.3 cannot be demonstrated at k=10 no matter how many pairs are collected. That
is a measurement problem, not a modelling one, and it is why this module reports
recall@1/@3/@5 and MRR as well.

The headroom is real at tighter k: only 11 of 38 positives rank first. That
matters beyond the metric — the agent is shown the top-k and cites from it, so
pulling the true catalyst from rank 3 to rank 1 changes what it quotes even when
recall@10 is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

from swing.store.session import connect

TARGET_PAIRS = 200          # agent-plan.md 5.3
KS = (1, 3, 5, 10)


@dataclass(frozen=True, slots=True)
class CatalystPair:
    swing_id: int
    ticker: str
    positive_cluster_id: int
    positive_rank: int
    negative_cluster_ids: list[int]

    @property
    def n_negatives(self) -> int:
        return len(self.negative_cluster_ids)


def pairs() -> list[CatalystPair]:
    """Every annotated swing whose true catalyst resolves to a pre-move cluster.

    Positives come from `annotations.true_article_ids` resolved THROUGH articles,
    not a stored cluster id: clusters are rebuilt whenever ranking is retuned, so
    a pinned id would rot (harness.recall_at_k carries the same warning).
    """
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT a.swing_id, s.ticker,
                   (SELECT c.id FROM clusters c
                    JOIN cluster_members m ON m.cluster_id = c.id
                    WHERE c.swing_id = a.swing_id AND c.timing = 'pre_move'
                      AND m.article_id = ANY(a.true_article_ids)
                    ORDER BY c.rank LIMIT 1) AS pos_id,
                   (SELECT min(c.rank) FROM clusters c
                    JOIN cluster_members m ON m.cluster_id = c.id
                    WHERE c.swing_id = a.swing_id AND c.timing = 'pre_move'
                      AND m.article_id = ANY(a.true_article_ids)) AS pos_rank
            FROM annotations a JOIN swings s ON s.id = a.swing_id
            WHERE a.blind AND NOT a.no_catalyst
              AND coalesce(cardinality(a.true_article_ids), 0) > 0
            ORDER BY a.swing_id
            """).fetchall()

        out: list[CatalystPair] = []
        for r in rows:
            if r["pos_id"] is None:
                continue        # catalyst is in the corpus but never clustered
            negs = conn.execute(
                # Hard negatives: the OTHER clusters retrieved for the same
                # swing. They are same-ticker, same-window and plausible, which
                # is what makes them hard — random articles teach nothing.
                "SELECT id FROM clusters WHERE swing_id = %s AND timing = 'pre_move' "
                "AND id <> %s ORDER BY rank", (r["swing_id"], r["pos_id"])).fetchall()
            out.append(CatalystPair(
                swing_id=r["swing_id"], ticker=r["ticker"],
                positive_cluster_id=r["pos_id"], positive_rank=r["pos_rank"],
                negative_cluster_ids=[n["id"] for n in negs]))
    return out


def baseline_ranking(rows: list[CatalystPair]) -> dict[str, float]:
    """Where the off-the-shelf embedder already puts the true catalyst."""
    if not rows:
        return {}
    ranks = [p.positive_rank for p in rows]
    out = {f"recall@{k}": sum(1 for r in ranks if r <= k) / len(ranks) for k in KS}
    out["MRR"] = sum(1.0 / r for r in ranks) / len(ranks)
    return out


def readiness() -> dict[str, object]:
    rows = pairs()
    metrics = baseline_ranking(rows)
    return {
        "pairs": len(rows),
        "target": TARGET_PAIRS,
        "short_by": max(0, TARGET_PAIRS - len(rows)),
        "hard_negatives": sum(p.n_negatives for p in rows),
        "baseline": metrics,
        # If the baseline is perfect at a given k, fine-tuning cannot be shown to
        # beat it there, however many pairs are collected.
        "saturated_at": [k for k in KS if metrics.get(f"recall@{k}") == 1.0],
    }


def main() -> int:
    state = readiness()
    print(f"  pairs: {state['pairs']} of {state['target']} "
          f"(short by {state['short_by']})   hard negatives: {state['hard_negatives']}")
    print("  off-the-shelf baseline:")
    for name, value in state["baseline"].items():
        print(f"    {name:<12} {value:.3f}")
    if state["saturated_at"]:
        ks = ", ".join(f"k={k}" for k in state["saturated_at"])
        print(f"\n  ⚠️ baseline is SATURATED at {ks}: nothing can beat 1.000 there, so "
              "Gate 5 for 5.3\n     cannot be demonstrated at that k. Judge on recall@1 "
              "and MRR, which have headroom.")
    if state["short_by"]:
        print(f"\n  BLOCKED: {state['short_by']} more annotated catalysts needed before "
              "fine-tuning is worth attempting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
