"""Populate `abstention precision` — the one gate metric that has never run.

`harness.abstention_precision` asks: of the swings the agent called
`unexplained`, what share truly had no catalyst? It reads `n/a` because it needs
the LATEST PRODUCTION attribution on an annotated swing to be `unexplained`, and
no `no_catalyst` swing has ever had a production attribution at all.

    .venv/bin/python -m swing.eval.abstention

The query is correct — unlike recall@10, there is no filter bug here. It is
starved, and the fix is to attribute exactly the swings a human marked as having
no catalyst. Those are the cases where abstaining is the right answer, so they
are also where a failure to abstain is most informative.

⚠️ This spends metered Gemini quota (free tier: 20/day, 5/minute), the same pool
Gate 4 draws on. One request per case, no retries, and it stops the moment the
API itself says the day is spent — see llm.batch_mode.
"""
from __future__ import annotations

from swing.common import logging as log

logger = log.get("eval.abstention")

# ⚠️ Raised from 2 once batch_mode() stopped retrying: a failure now costs one
# request instead of three, and the real "stop now" signal is the daily cap,
# which is detected outright. Two was tuned when a failure was expensive, and it
# ended runs on a pair of 503s with most of the day still unspent.
MAX_CONSECUTIVE_FAILURES = 4


def pending() -> list[int]:
    """Annotated no-catalyst swings with clusters but no production attribution."""
    from swing.store.session import connect

    with connect() as conn:
        return [r["swing_id"] for r in conn.execute(
            """
            SELECT n.swing_id FROM annotations n
            WHERE n.no_catalyst
              AND EXISTS (SELECT 1 FROM clusters c WHERE c.swing_id = n.swing_id)
              AND NOT EXISTS (SELECT 1 FROM attributions a
                              WHERE a.swing_id = n.swing_id AND a.run_kind = 'production')
            ORDER BY n.swing_id
            """).fetchall()]


def run(limit: int | None = None) -> dict:
    """Attribute the pending no-catalyst swings. Returns what happened."""
    from swing.agent.graph import attribute_swing
    from swing.agent.llm import batch_mode, daily_cap_reached

    targets = pending()
    if limit is not None:
        targets = targets[:limit]
    if not targets:
        return {"attributed": 0, "note": "no pending no-catalyst swings"}

    verdicts: dict[str, int] = {}
    failures = 0
    # One request per case: see llm.batch_mode.
    with batch_mode():
        for swing_id in targets:
            out = attribute_swing(swing_id, run_kind="production", persist=True)
            if out.get("verdict_reason") == "llm_error":
                if daily_cap_reached():
                    logger.warning("abstention: daily quota exhausted; "
                                   "%d still pending for tomorrow", len(targets))
                    break
                failures += 1
                logger.warning("abstention: model call failed for swing %s (%d in a row)",
                               swing_id, failures)
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.warning("abstention: stopping after %d consecutive failures", failures)
                    break
                continue
            failures = 0
            attr = out.get("attribution")
            verdict = attr.verdict if attr else "error"
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
            logger.info("abstention: swing %s -> %s", swing_id, verdict)
    return {"attributed": sum(verdicts.values()), "verdicts": verdicts,
            "remaining": len(pending())}


def main() -> int:
    outstanding = pending()
    print(f"  {len(outstanding)} no-catalyst swings awaiting a production attribution")
    if not outstanding:
        print("  nothing to do")
    else:
        result = run()
        print(f"  attributed {result['attributed']}: {result.get('verdicts', {})}")
        print(f"  still pending: {result['remaining']}")

    from swing.eval.harness import abstention_precision

    print("\n  " + abstention_precision().line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
