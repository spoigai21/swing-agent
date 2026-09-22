"""Does the agent know when it is right?

Every candidate the model returns carries three self-assessments — `confidence`,
`direction_consistent` and `magnitude_plausible`. Two of them were never checked
against reality and one was never read by any code at all:

    confidence           shown to the user, never validated
    direction_consistent trusted by enforce_abstention to keep a candidate
    magnitude_plausible  read by nothing, anywhere

That is the cheapest learning signal in the system and it was being thrown away.
Scoring them against the human annotations costs no quota and no model calls,
and it answers a question the product makes implicitly on every answer: when it
says "high confidence", how often is it actually right?

    .venv/bin/python -m swing.eval.calibration     (or: swing calibration)

⚠️ This MEASURES. It does not retrain or tune anything. A calibration that is
itself fitted to the annotations would tell you nothing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import pairwise

from swing.common import logging as log

logger = log.get("eval.calibration")

CONFIDENCE_ORDER = ("high", "medium", "low")


@dataclass
class Bucket:
    label: str
    n: int = 0
    hits: int = 0
    swing_ids: list[int] = field(default_factory=list)

    @property
    def accuracy(self) -> float | None:
        return self.hits / self.n if self.n else None

    def bounds(self) -> tuple[float, float] | None:
        """Wilson interval — usable at the small n this will run at for a while."""
        if not self.n:
            return None
        import math

        z, p, n = 1.96, self.hits / self.n, self.n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return max(0.0, centre - half), min(1.0, centre + half)

    def line(self) -> str:
        if not self.n:
            return f"  {self.label:<22} no cases yet"
        lo, hi = self.bounds()
        return (f"  {self.label:<22} {self.accuracy:.2f}  n={self.n:<4} "
                f"95% CI [{lo:.2f}, {hi:.2f}]")


def _payload(raw) -> dict:
    return raw if isinstance(raw, dict) else json.loads(raw)


def scored_cases() -> list[dict]:
    """Annotated swings whose latest production answer named a catalyst.

    Same join as `harness.attribution_accuracy`, so a case counted right here is
    counted right there too.
    """
    from swing.eval.harness import top_candidate_hit
    from swing.store.session import connect

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
        cases = []
        for r in rows:
            payload = _payload(r["payload"])
            cands = payload.get("candidates") or []
            if not cands:
                continue                      # abstained: scored by another metric
            top = cands[0]
            cited = [e["cluster_id"] for e in (top.get("evidence") or [])]
            members: dict[int, set[int]] = {}
            if cited:
                for m in conn.execute(
                        "SELECT cluster_id, article_id FROM cluster_members "
                        "WHERE cluster_id = ANY(%s)", (cited,)).fetchall():
                    members.setdefault(m["cluster_id"], set()).add(m["article_id"])
            cases.append({
                "swing_id": r["swing_id"],
                "confidence": (top.get("confidence") or "unstated").lower(),
                "direction_consistent": top.get("direction_consistent"),
                "magnitude_plausible": top.get("magnitude_plausible"),
                "correct": bool(top_candidate_hit(payload, r["true_cluster_id"],
                                                  r["true_article_ids"], members)),
            })
    return cases


def by_confidence(cases: list[dict] | None = None) -> list[Bucket]:
    cases = scored_cases() if cases is None else cases
    buckets = {c: Bucket(c) for c in CONFIDENCE_ORDER}
    for case in cases:
        b = buckets.setdefault(case["confidence"], Bucket(case["confidence"]))
        b.n += 1
        b.hits += case["correct"]
        b.swing_ids.append(case["swing_id"])
    return [buckets[k] for k in CONFIDENCE_ORDER if k in buckets] + [
        b for k, b in buckets.items() if k not in CONFIDENCE_ORDER]


def by_flag(flag: str, cases: list[dict] | None = None) -> list[Bucket]:
    """Accuracy when the model asserted a flag versus when it denied it."""
    cases = scored_cases() if cases is None else cases
    out = {True: Bucket(f"{flag}=true"), False: Bucket(f"{flag}=false")}
    for case in cases:
        v = case.get(flag)
        if v is None:
            continue
        b = out[bool(v)]
        b.n += 1
        b.hits += case["correct"]
    return [out[True], out[False]]


def is_monotonic(buckets: list[Bucket]) -> bool | None:
    """Is 'high' at least as accurate as 'medium', and 'medium' as 'low'?

    The one property a confidence label has to have to be worth printing. None
    when fewer than two buckets have cases — undecidable, not satisfied.
    """
    seen = [b for b in buckets if b.n and b.label in CONFIDENCE_ORDER]
    if len(seen) < 2:
        return None
    order = {c: i for i, c in enumerate(CONFIDENCE_ORDER)}
    seen.sort(key=lambda b: order[b.label])
    return all(a.accuracy >= b.accuracy - 1e-9 for a, b in pairwise(seen))


def flag_values(flag: str) -> dict:
    """How often the model has EVER asserted each value of a self-assessment.

    ⚠️ Separate from the scored cases on purpose. A flag that only ever takes
    one value carries no information no matter how accurate the answers are,
    and that is checkable from every candidate ever returned rather than only
    the annotated handful. As of 2026-09-22 `direction_consistent` and
    `magnitude_plausible` were `true` in 7 of 7 candidates — so the
    `direction_consistent` test in `enforce_abstention` has never once rejected
    a candidate. A guard that has never fired is an assumption.
    """
    import json as _json
    from collections import Counter

    from swing.store.session import connect

    seen: Counter = Counter()
    with connect() as conn:
        for r in conn.execute("SELECT payload FROM attributions").fetchall():
            payload = r["payload"] if isinstance(r["payload"], dict) else _json.loads(r["payload"])
            for cand in payload.get("candidates") or []:
                seen[cand.get(flag)] += 1
    return dict(seen)


def flag_report(flag: str) -> str:
    seen = flag_values(flag)
    total = sum(seen.values())
    if not total:
        return f"  {flag:<22} never returned"
    shown = ", ".join(f"{k}={v}" for k, v in sorted(seen.items(), key=lambda kv: str(kv[0])))
    if len(seen) == 1:
        only = next(iter(seen))
        return (f"  {flag:<22} {shown}  ⚠ always {only} in {total} candidate(s) — "
                f"carries no information")
    return f"  {flag:<22} {shown}"


def report() -> str:
    cases = scored_cases()
    lines = [f"\nConfidence calibration — {len(cases)} scored case(s)\n"]
    if not cases:
        return ("\nNo scored cases yet. Calibration needs annotated swings that the "
                "agent answered with a catalyst; run `swing annotate` and the "
                "accuracy backlog first.")
    conf = by_confidence(cases)
    lines += [b.line() for b in conf]
    mono = is_monotonic(conf)
    if mono is None:
        lines.append("\n  Only one confidence level has cases — nothing to compare yet.")
    elif mono:
        lines.append("\n  ✓ Higher confidence is at least as accurate as lower.")
    else:
        lines.append("\n  ⚠ NOT monotonic: a lower confidence level scored better than a "
                     "higher one.\n    Until that reverses, the label is decoration and "
                     "should not be shown as if it means something.")
    lines.append("\nSelf-assessment flags, across every candidate ever returned:")
    for flag in ("direction_consistent", "magnitude_plausible"):
        lines.append(flag_report(flag))
    lines.append("\nAccuracy when each flag was asserted (scored cases only):")
    for flag in ("direction_consistent", "magnitude_plausible"):
        lines += [b.line() for b in by_flag(flag, cases) if b.n]
    lines.append("\n  ⚠ Small samples. A bucket under ~10 cases cannot separate 0.9 from "
                 "0.5; read the intervals, not the point estimates.")
    return "\n".join(lines)


def main() -> int:
    print(report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
