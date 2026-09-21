"""The five metrics. agent-plan.md 4.2.

| metric               | definition                                    | target |
|----------------------|-----------------------------------------------|--------|
| catalyst coverage    | labelled moves whose catalyst is in the corpus | >= 0.80 |
| recall@10 (covered)  | of those, true catalyst cluster in top 10     | >= 0.85 |
| retrieval recall@10  | overall: coverage x recall-given-coverage     | >= 0.68 |
| attribution accuracy | top candidate matches the annotation          | > 0.70 |
| abstention precision | of `unexplained`, share that truly had none   | > 0.80 |
| confabulation rate   | of no-catalyst cases, share explained anyway  | < 0.10 |
| citation validity    | cited clusters existing in the retrieved set  | 1.00   |

⚠️ recall@10 is computed ONLY over blind annotations. Computing it over
assisted rows is circular: those labels were PICKED from the retrieved list, so
recall is 1.0 by construction and the metric measures nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

from swing.store.session import connect


@dataclass(slots=True)
class Metric:
    name: str
    value: float | None
    n: int
    target: str
    passing: bool | None

    def line(self) -> str:
        v = "n/a " if self.value is None else f"{self.value:.3f}"
        mark = "" if self.passing is None else ("PASS" if self.passing else "FAIL")
        return f"  {self.name:<24}{v:>7}  n={self.n:<5} target {self.target:<8} {mark}"


def recall_at_k(k: int = 10) -> Metric:
    """⚠️ The label is resolved through its ARTICLES, not a stored cluster id.

    Clusters are rebuilt every time ranking is retuned, so a stored cluster id
    pins the label to the ranking that existed when you annotated. Matching the
    labelled articles against the clusters that exist NOW makes recall move when
    the ranking does, and a re-cluster that merges or splits the story still
    scores (best rank wins).
    """
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT a.swing_id,
                   (SELECT min(c.rank) FROM clusters c
                    JOIN cluster_members m ON m.cluster_id = c.id
                    WHERE c.swing_id = a.swing_id AND c.timing = 'pre_move'
                      AND m.article_id = ANY(a.true_article_ids)) AS rank
            FROM annotations a
            WHERE a.blind AND NOT a.no_catalyst
            """).fetchall()
    # Re-baselined (CODEBASE-PLAN 16.13). This number is the PRODUCT of two
    # independent things -- whether the catalyst is in the corpus at all, and
    # whether retrieval ranks it once it is -- so its target is the product of
    # their targets, 0.80 x 0.85 = 0.68, never a number picked to pass.
    if not rows:
        return Metric(f"recall@{k} (blind)", None, 0, ">= 0.68", None)
    hits = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= k)
    v = hits / len(rows)
    return Metric(f"recall@{k} (blind)", v, len(rows), ">= 0.68", v >= 0.68)


def catalyst_coverage() -> Metric:
    """Of labelled moves, the share whose catalyst article is in the corpus AT ALL.

    A DATA metric, not a system one. An annotator records `true_catalyst` as free
    text even when no article for it was ever collected, leaving
    `true_article_ids` empty -- so an empty list means "we know the cause and do
    not hold the story", which no amount of ranking work can fix.

    Kept separate from recall so the two cannot mask each other: a source outage
    shows up here as falling coverage instead of silently excusing bad ranking,
    and retuning weights cannot improve this number at all.
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT coalesce(cardinality(true_article_ids), 0) AS n_articles "
            "FROM annotations WHERE blind AND NOT no_catalyst").fetchall()
    if not rows:
        return Metric("catalyst coverage", None, 0, ">= 0.80", None)
    have = sum(1 for r in rows if r["n_articles"])
    v = have / len(rows)
    return Metric("catalyst coverage", v, len(rows), ">= 0.80", v >= 0.80)


def recall_at_k_covered(k: int = 10) -> Metric:
    """recall@k over ONLY the moves whose catalyst we actually hold.

    This is the retrieval engine's own score, with the news archive's gaps taken
    out of the denominator. It is the demanding one: when the evidence is
    present and retrieval still fails to rank it, that is a real defect with
    nowhere to hide.
    """
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT a.swing_id,
                   (SELECT min(c.rank) FROM clusters c
                    JOIN cluster_members m ON m.cluster_id = c.id
                    WHERE c.swing_id = a.swing_id AND c.timing = 'pre_move'
                      AND m.article_id = ANY(a.true_article_ids)) AS rank
            FROM annotations a
            WHERE a.blind AND NOT a.no_catalyst
              AND coalesce(cardinality(a.true_article_ids), 0) > 0
            """).fetchall()
    if not rows:
        return Metric(f"recall@{k} (covered)", None, 0, ">= 0.85", None)
    hits = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= k)
    v = hits / len(rows)
    return Metric(f"recall@{k} (covered)", v, len(rows), ">= 0.85", v >= 0.85)


def _top_cited(payload: dict | None) -> set[int]:
    cands = (payload or {}).get("candidates") or []
    return {e["cluster_id"] for c in cands[:1] for e in (c.get("evidence") or [])
            if e.get("cluster_id") is not None}


def top_candidate_hit(payload: dict | None, true_cluster_id: int | None,
                      true_article_ids: list[int] | None,
                      members: dict[int, set[int]]) -> bool:
    """Does the top candidate cite the annotated story?

    Matched by cluster id OR by a shared article, so a labelled cluster that was
    replaced by a rebuild still matches an attribution citing its successor.
    """
    cited = _top_cited(payload)
    if true_cluster_id is not None and true_cluster_id in cited:
        return True
    truth = set(true_article_ids or [])
    return any(members.get(cid, set()) & truth for cid in cited)


def attribution_accuracy() -> Metric:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT a.swing_id, a.true_cluster_id, a.true_article_ids, at.payload
            FROM annotations a
            JOIN LATERAL (SELECT payload FROM attributions
                          WHERE swing_id=a.swing_id AND run_kind='production'
                          ORDER BY created_at DESC LIMIT 1) at ON true
            WHERE NOT a.no_catalyst
              AND (a.true_article_ids IS NOT NULL OR a.true_cluster_id IS NOT NULL)
            """).fetchall()
        cited = sorted(set().union(*(_top_cited(r["payload"]) for r in rows)))
        members: dict[int, set[int]] = {}
        if cited:
            for m in conn.execute(
                    "SELECT cluster_id, article_id FROM cluster_members "
                    "WHERE cluster_id = ANY(%s)", (cited,)).fetchall():
                members.setdefault(m["cluster_id"], set()).add(m["article_id"])
    if not rows:
        return Metric("attribution accuracy", None, 0, "> 0.70", None)
    hit = sum(1 for r in rows if top_candidate_hit(
        r["payload"], r["true_cluster_id"], r["true_article_ids"], members))
    v = hit / len(rows)
    return Metric("attribution accuracy", v, len(rows), "> 0.70", v > 0.70)


def abstention_precision() -> Metric:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT a.no_catalyst, at.verdict
            FROM annotations a
            JOIN LATERAL (SELECT verdict FROM attributions
                          WHERE swing_id=a.swing_id AND run_kind='production'
                          ORDER BY created_at DESC LIMIT 1) at ON true
            WHERE at.verdict = 'unexplained'
            """).fetchall()
    if not rows:
        return Metric("abstention precision", None, 0, "> 0.80", None)
    v = sum(1 for r in rows if r["no_catalyst"]) / len(rows)
    return Metric("abstention precision", v, len(rows), "> 0.80", v > 0.80)


def confabulation_rate(run_kind: str = "placebo") -> Metric:
    """Of cases with NO true catalyst, the share that got a confident story.

    The metric that matters most: a system that explains 90% correctly and
    invents the other 10% is worse than one explaining 70% and abstaining,
    because you cannot tell which bucket an answer is in.

    ⚠️ Filtered to the current (model, prompt, config, eval-design) tuple.
    Pooling across versions is exactly what agent-plan.md 3.1b exists to
    prevent, and an eval-design change alters what the number MEANS.
    """
    from swing.common.versioning import config_hash, eval_hash, model_id, prompt_version

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT verdict FROM attributions
            WHERE run_kind=%s AND model_id=%s AND prompt_version=%s
              AND config_hash=%s AND coalesce(verdict_reason,'') LIKE %s
            """,
            (run_kind, model_id(), prompt_version(), config_hash(),
             f"placebo:{eval_hash()}%"),
        ).fetchall()
    if not rows:
        return Metric(f"confabulation ({run_kind})", None, 0, "< 0.10", None)
    bad = sum(1 for r in rows if r["verdict"] != "unexplained")
    v = bad / len(rows)
    return Metric(f"confabulation ({run_kind})", v, len(rows), "< 0.10", v < 0.10)


def citation_validity() -> Metric:
    """Every cited cluster_id must exist in what we actually PASSED the model.

    Validated against `shown_cluster_ids`, not the swing's own clusters: a
    placebo run is shown the DONOR's clusters, and checking against the swing's
    own set scored every valid citation invalid (the metric read 0.000).
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, swing_id, payload, shown_cluster_ids FROM attributions"
        ).fetchall()
    total = ok = 0
    for r in rows:
        allowed = set(r["shown_cluster_ids"] or [])
        if not allowed:
            continue          # pre-dates the column; cannot be judged
        for c in (r["payload"] or {}).get("candidates") or []:
            for e in c.get("evidence") or []:
                total += 1
                if e.get("cluster_id") in allowed:
                    ok += 1
    if total == 0:
        return Metric("citation validity", None, 0, "= 1.00", None)
    v = ok / total
    return Metric("citation validity", v, total, "= 1.00", v == 1.0)


def report() -> list[Metric]:
    return [catalyst_coverage(), recall_at_k_covered(), recall_at_k(),
            attribution_accuracy(), abstention_precision(),
            confabulation_rate(), citation_validity()]


def print_report() -> None:
    print(f"  {'metric':<24}{'value':>7}  {'n':<7} {'target':<15} status")
    print("  " + "-" * 66)
    for m in report():
        print(m.line())
