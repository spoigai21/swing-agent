"""Fit the ranking weights instead of guessing them.

`w_semantic`, `w_timing`, `w_tier` and `w_related_penalty` were hand-set in
thresholds.yaml and never fitted, while a 55-move answer key and a walk-forward
split helper sat unused next to them.

Nothing here calls a model or an embedder. Every component score
(`semantic_score`, `timing_score`, `best_tier`, `novelty_score`) is already
persisted per cluster, so re-scoring a candidate weight vector is arithmetic
over rows the database already holds — a full grid runs in seconds and spends
no quota.

⚠️ TIME-FORWARD, ALWAYS. Weights chosen on the swings they are then scored on
would report a number that means nothing. Fitting happens on the earlier swings
and the reported figure comes from the later ones, once.

⚠️ APPLYING THE RESULT CHANGES `config_hash`, which correctly resets the Gate 4
placebo count. `fit()` only reports; writing thresholds.yaml is a separate,
explicit step.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import product

from swing.common import logging as log

logger = log.get("models.weights")

#: Coarse by design. A finer grid on ~40 swings fits the noise, not the ranking.
GRID = {
    "w_semantic": (0.5, 1.0, 1.5, 2.0),
    "w_timing": (0.5, 1.0, 1.5, 2.0),
    "w_tier": (0.0, 0.25, 0.5, 1.0),
    "w_related_penalty": (0.0, 0.5, 1.0),
}
TOP_K = 10


@dataclass(frozen=True, slots=True)
class Cluster:
    semantic: float
    timing: float
    tier: int
    novelty: float
    related_only: bool
    is_true: bool

    def score(self, w: dict[str, float]) -> float:
        return (w["w_semantic"] * self.semantic
                + w["w_timing"] * self.timing
                + w["w_tier"] * (5 - self.tier)
                + w.get("w_novelty", 0.0) * self.novelty
                - (w["w_related_penalty"] if self.related_only else 0.0))


@dataclass
class Case:
    swing_id: int
    d: date
    clusters: list[Cluster]

    @property
    def scorable(self) -> bool:
        """A case with no labelled cluster cannot reward any weight vector."""
        return any(c.is_true for c in self.clusters)


def load_cases() -> list[Case]:
    """Blind, labelled swings with their pre-move clusters and component scores."""
    from swing.store.session import connect

    sql = """
    SELECT s.id AS swing_id, s.d, s.ticker,
           c.id AS cluster_id, c.semantic_score, c.timing_score, c.best_tier,
           c.novelty_score,
           EXISTS (SELECT 1 FROM cluster_members m JOIN articles ar ON ar.id = m.article_id
                   WHERE m.cluster_id = c.id AND s.ticker = ANY(ar.tickers)) AS own_news,
           EXISTS (SELECT 1 FROM cluster_members m
                   WHERE m.cluster_id = c.id AND m.article_id = ANY(a.true_article_ids)) AS is_true
    FROM annotations a
    JOIN swings s ON s.id = a.swing_id
    JOIN clusters c ON c.swing_id = a.swing_id AND c.timing = 'pre_move'
    WHERE a.blind AND NOT a.no_catalyst AND a.true_article_ids IS NOT NULL
    ORDER BY s.d, s.id
    """
    cases: dict[int, Case] = {}
    with connect() as conn:
        for r in conn.execute(sql).fetchall():
            case = cases.setdefault(r["swing_id"], Case(r["swing_id"], r["d"], []))
            case.clusters.append(Cluster(
                semantic=float(r["semantic_score"] or 0.5),
                timing=float(r["timing_score"] or 0.0),
                tier=int(r["best_tier"] or 4),
                novelty=float(r["novelty_score"] or 0.0),
                related_only=not r["own_news"],
                is_true=bool(r["is_true"]),
            ))
    return [c for c in cases.values() if c.scorable]


def recall_at_k(cases: list[Case], w: dict[str, float], k: int = TOP_K) -> float:
    """Share of cases whose labelled cluster lands in the top k under `w`."""
    if not cases:
        return 0.0
    hits = 0
    for case in cases:
        ranked = sorted(case.clusters, key=lambda c: -c.score(w))
        hits += any(c.is_true for c in ranked[:k])
    return hits / len(cases)


def current_weights() -> dict[str, float]:
    from swing.ingest.config import thresholds

    cfg = thresholds()["ranking"]
    return {k: float(cfg.get(k, 0.0)) for k in
            ("w_semantic", "w_timing", "w_tier", "w_novelty", "w_related_penalty")}


def grid_search(cases: list[Case], k: int = TOP_K) -> tuple[dict[str, float], float]:
    """Best weights on `cases`. Ties break toward the weights already in use."""
    base = current_weights()
    best_w, best_score = dict(base), recall_at_k(cases, base, k)
    for combo in product(*GRID.values()):
        w = dict(base) | dict(zip(GRID.keys(), combo, strict=True))
        score = recall_at_k(cases, w, k)
        if score > best_score:
            best_w, best_score = w, score
    return best_w, best_score


def fit(k: int = TOP_K, train_frac: float = 0.7) -> dict:
    """Fit on the earlier swings, report on the later ones. Never writes."""
    cases = load_cases()
    if len(cases) < 8:
        return {"error": f"only {len(cases)} scorable cases; too few to split"}

    cases.sort(key=lambda c: (c.d, c.swing_id))
    cut = max(4, int(len(cases) * train_frac))
    train, test = cases[:cut], cases[cut:]

    # ⚠️ The split helper exists so no file can shuffle. Assert it here too.
    from swing.models.splits import assert_time_forward

    assert_time_forward(train, test, "d")

    base = current_weights()
    fitted, train_score = grid_search(train, k)
    return {
        "n_cases": len(cases), "n_train": len(train), "n_test": len(test),
        "cutoff": str(test[0].d),
        "current": base,
        "fitted": fitted,
        "train_current": recall_at_k(train, base, k),
        "train_fitted": train_score,
        "test_current": recall_at_k(test, base, k),
        "test_fitted": recall_at_k(test, fitted, k),
        "changed": {k2: (base[k2], fitted[k2]) for k2 in fitted if base[k2] != fitted[k2]},
    }


def combos() -> int:
    n = 1
    for values in GRID.values():
        n *= len(values)
    return n


def verdict(result: dict) -> str:
    """Did fitting earn its place? One test case is worth 1/n_test of recall.

    ⚠️ Two hurdles, not one. The grid tries ~200 weight vectors, so the best of
    them beats the default on the TRAINING cases by construction; that number is
    not evidence. And a held-out gain worth a case or two, on a dozen cases, is
    the same size as the noise — §16.25 recorded three separate versions of a
    Phase 5 verdict rule that each blessed a win finer than the instrument.
    """
    if "error" in result:
        return result["error"]
    gain = result["test_fitted"] - result["test_current"]
    resolution = 1.0 / result["n_test"]
    if not result["changed"]:
        return "The hand-set weights already win the grid. Nothing to change."
    if gain <= 0:
        return (f"Fitted weights do NOT beat the current ones on held-out data "
                f"({result['test_fitted']:.3f} vs {result['test_current']:.3f}). Keep the current ones.")
    if gain < resolution:
        return (f"Gain {gain:+.3f} is smaller than one test case ({resolution:.3f}), "
                "so it is inside the resolution of the measurement. Keep the current ones.")
    cases_won = round(gain * result["n_test"])
    if gain < 2 * resolution:
        return (f"Fitted weights win by {gain:+.3f} on held-out data — {cases_won} case "
                f"of {result['n_test']}, from a search over {combos()} weight vectors. "
                "Suggestive, not decisive: label more swings before applying it.")
    return (f"Fitted weights beat the current ones by {gain:+.3f} on held-out data "
            f"({cases_won} of {result['n_test']} cases, search width {combos()}). "
            "Worth applying, and worth re-checking once more swings are labelled.")


def report(k: int = TOP_K) -> str:
    r = fit(k=k)
    if "error" in r:
        return f"\n{r['error']}"
    lines = [
        (f"\nRanking weights — {r['n_cases']} labelled swings "
         f"({r['n_train']} train, {r['n_test']} held out from {r['cutoff']})\n"),
        f"  {'weight':<20}{'current':>10}{'fitted':>10}",
    ]
    for key in ("w_semantic", "w_timing", "w_tier", "w_related_penalty"):
        mark = "  <-" if r["current"][key] != r["fitted"][key] else ""
        lines.append(f"  {key:<20}{r['current'][key]:>10}{r['fitted'][key]:>10}{mark}")
    lines += [
        "",
        f"  recall@{k} train    current {r['train_current']:.3f}   fitted {r['train_fitted']:.3f}",
        f"  recall@{k} held out current {r['test_current']:.3f}   fitted {r['test_fitted']:.3f}",
        "",
        f"  {verdict(r)}",
        "",
        "  ⚠ Applying these edits thresholds.yaml, which changes config_hash and",
        "    correctly resets the Gate 4 placebo count. Do it deliberately.",
    ]
    return "\n".join(lines)


def main() -> int:
    print(report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
