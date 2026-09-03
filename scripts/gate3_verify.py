#!/usr/bin/env python
"""Gate 3. agent-plan.md: 10 real swings + 5 synthetic no-news cases.

ALL 5 synthetic cases must return `unexplained`. A synthetic case is a real
price move paired with articles from a different week for the same ticker, so it
cannot be passed by recalling what actually happened.
"""
from __future__ import annotations

import argparse

from swing.agent.graph import attribute_swing
from swing.eval.cache import cached_clusters
from swing.eval.placebo import build_cases
from swing.store.session import connect


def real_swings(n: int) -> list[dict]:
    with connect() as conn:
        return conn.execute(
            """
            SELECT s.id, s.ticker, s.d, s.residual_z, s.earnings_mode, s.swing_type,
                   count(*) FILTER (WHERE c.timing='pre_move') AS pre
            FROM swings s JOIN clusters c ON c.swing_id=s.id
            GROUP BY s.id, s.ticker, s.d, s.residual_z, s.earnings_mode, s.swing_type
            HAVING count(*) FILTER (WHERE c.timing='pre_move') >= 3
            ORDER BY abs(s.residual_z) DESC LIMIT %s
            """,
            (n,),
        ).fetchall()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=int, default=10)
    ap.add_argument("--synthetic", type=int, default=5)
    ap.add_argument("--persist", action="store_true")
    a = ap.parse_args()

    print("=" * 78)
    print(f"  PART 1 — {a.real} real swings")
    print("=" * 78)
    print(f"  {'ticker':<7}{'date':<12}{'z':>7}{'pre':>5}  {'verdict':<20}cand  top candidate")
    print("  " + "-" * 74)
    for s in real_swings(a.real):
        out = attribute_swing(s["id"], run_kind="eval", persist=a.persist)
        attr = out.get("attribution")
        top = attr.candidates[0].catalyst[:30] if attr and attr.candidates else "-"
        etype = f"[{attr.candidates[0].event_type}]" if attr and attr.candidates else ""
        print(f"  {s['ticker']:<7}{s['d']!s:<12}{float(s['residual_z']):>7.2f}"
              f"{s['pre']:>5}  {attr.verdict if attr else 'ERROR':<20}"
              f"{len(attr.candidates) if attr else 0:<6}{etype} {top}")

    print()
    print("=" * 78)
    print(f"  PART 2 — {a.synthetic} synthetic no-news cases (ALL must be `unexplained`)")
    print("=" * 78)
    cases = build_cases(a.synthetic, seed=7)
    if not cases:
        print("  no eligible swings — need more pre-move coverage")
        return 1
    passed = 0
    for c in cases:
        donor = cached_clusters(c.donor_swing_id)
        out = attribute_swing(c.swing_id, run_kind="placebo", persist=a.persist,
                              cluster_override=donor)
        attr = out.get("attribution")
        v = attr.verdict if attr else "ERROR"
        ok = v == "unexplained"
        passed += ok
        detail = "" if ok else f"  <-- CONFABULATED: {attr.candidates[0].catalyst[:44]}"
        print(f"  swing {c.swing_id:<5} {c.ticker:<6} donor {c.donor_swing_id:<5} "
              f"-> {v:<22}{'PASS' if ok else 'FAIL'}{detail}")
    print()
    print(f"  GATE 3: {passed}/{len(cases)} synthetic cases abstained "
          f"— {'PASS' if passed == len(cases) else 'FAIL'}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
