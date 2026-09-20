"""Feed `attribution accuracy` — the metric that says the answers are RIGHT.

Three metrics divide the agent's honesty between them:

    confabulation      does it invent when it has nothing?        0.000, n=34
    abstention         does it shrug when it does have something?  n=1
    attribution        when it explains, is the explanation right? n=3

The third is the one that says the product works, and it is the thinnest. A
perfect 3-of-3 is consistent with a true rate as low as 29%, well under its 0.70
target; **n=11 is the first perfect run that establishes >0.70 at 95%
confidence**. There are 36 annotated swings with a known catalyst that have
never been asked, so the shortfall is spent quota, not missing labels.

    .venv/bin/python -m swing.eval.accuracy

⚠️ Spends metered Gemini quota (free tier: 20/day), the same pool Gate 4 draws
on. Gate 4's own bar already passes at n=34, so this is the better use of it.
One request per case, no retries, and it stops the moment the API itself
says the day is spent — see llm.batch_mode.
"""
from __future__ import annotations

from swing.common import logging as log

logger = log.get("eval.accuracy")

# ⚠️ Raised from 2 once batch_mode() stopped retrying: a failure now costs one
# request instead of three, and the real "stop now" signal is the daily cap,
# which is detected outright. Two was tuned when a failure was expensive, and it
# ended runs on a pair of 503s with most of the day still unspent.
MAX_CONSECUTIVE_FAILURES = 4


def pending() -> list[int]:
    """Annotated swings with a known catalyst and no production attribution."""
    from swing.store.session import connect

    with connect() as conn:
        return [r["swing_id"] for r in conn.execute(
            """
            SELECT n.swing_id FROM annotations n
            WHERE NOT n.no_catalyst
              AND coalesce(cardinality(n.true_article_ids), 0) > 0
              AND EXISTS (SELECT 1 FROM clusters c WHERE c.swing_id = n.swing_id)
              AND NOT EXISTS (SELECT 1 FROM attributions a
                              WHERE a.swing_id = n.swing_id AND a.run_kind = 'production')
            ORDER BY n.swing_id
            """).fetchall()]


def run(limit: int | None = None) -> dict:
    """Attribute pending swings so the accuracy metric has something to score."""
    from swing.agent.graph import attribute_swing
    from swing.agent.llm import batch_mode, daily_cap_reached

    targets = pending()
    if limit is not None:
        targets = targets[:limit]
    if not targets:
        return {"attributed": 0, "note": "no pending annotated swings"}

    verdicts: dict[str, int] = {}
    failures = 0
    # One request per case: see llm.batch_mode.
    with batch_mode():
        for swing_id in targets:
            out = attribute_swing(swing_id, run_kind="production", persist=True)
            if out.get("verdict_reason") == "llm_error":
                if daily_cap_reached():
                    logger.warning("accuracy: daily quota exhausted; "
                                   "%d still pending for tomorrow", len(targets))
                    break
                failures += 1
                logger.warning("accuracy: model call failed for swing %s (%d in a row)",
                               swing_id, failures)
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.warning("accuracy: stopping after %d consecutive failures", failures)
                    break
                continue
            failures = 0
            attr = out.get("attribution")
            verdict = attr.verdict if attr else "error"
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
            logger.info("accuracy: swing %s -> %s", swing_id, verdict)
    return {"attributed": sum(verdicts.values()), "verdicts": verdicts,
            "remaining": len(pending())}


def main() -> int:
    outstanding = pending()
    print(f"  {len(outstanding)} annotated swings with a known catalyst, never attributed")
    if outstanding:
        result = run()
        print(f"  attributed {result['attributed']}: {result.get('verdicts', {})}")
        print(f"  still pending: {result['remaining']}")

    from swing.eval.harness import attribution_accuracy

    print("\n  " + attribution_accuracy().line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
