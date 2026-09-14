"""Daily post-close batch. agent-plan.md 6.1.

Order matters: prices -> factors -> swings -> intraday capture -> clusters ->
attribution, with SECTOR ENTITIES FIRST so a stock attribution can reference an
already-computed sector story (0.1b).

⚠️ Free-tier LLM quota is 20 requests/day per model, so the attribution stage is
capped by default. Everything before it is unmetered and always runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from swing.common import logging as log

logger = log.get("interface.batch")


@dataclass(slots=True)
class BatchResult:
    started_at: datetime
    bars_written: int = 0
    factor_rows: int = 0
    swings_found: int = 0
    onsets_fixed: int = 0
    clusters_built: int = 0
    attributed: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"bars={self.bars_written}", f"factors={self.factor_rows}",
            f"swings={self.swings_found}", f"onsets={self.onsets_fixed}",
            f"clusters={self.clusters_built}",
        ]
        if self.attributed:
            parts.append("attributed=" + ",".join(f"{k}:{v}" for k, v in
                                                  sorted(self.attributed.items())))
        if self.errors:
            parts.append(f"errors={len(self.errors)}")
        return "  ".join(parts)


def run(day: date | None = None, attribution_limit: int = 15,
        skip_prices: bool = False, skip_attribution: bool = False,
        attribute_since: date | None = None) -> BatchResult:
    """One end-to-end daily run.

    Each stage is wrapped: a failure in one must not abort the rest, because a
    price-vendor outage should never stop the news pipeline. `attribute_since`
    limits the (quota-spending) attribution stage to recent swings, so a
    scheduled run explains today's moves rather than working through history.
    """
    from swing.analysis.factors import rebuild_all
    from swing.analysis.retrieval import build_all
    from swing.analysis.swings import backfill_onsets, detect_all
    from swing.ingest.normalize import normalize_all
    from swing.ingest.prices import backfill_daily

    res = BatchResult(started_at=datetime.now(UTC))
    day = day or datetime.now(UTC).date()

    def stage(name: str, fn):
        try:
            return fn()
        except Exception as exc:
            logger.exception("batch stage %s failed", name)
            res.errors.append(f"{name}: {type(exc).__name__}: {exc}")
            return None

    if not skip_prices:
        # 5 days rather than 1: covers weekends, holidays and a missed run.
        written = stage("prices", lambda: backfill_daily(years=1))
        res.bars_written = sum((written or {}).values())

    got = stage("normalize", lambda: normalize_all())
    if got:
        logger.info("normalized %s", got)

    factors = stage("factors", rebuild_all)
    res.factor_rows = sum((factors or {}).values())

    swings = stage("detect", lambda: detect_all(capture_intraday=False))
    res.swings_found = sum((swings or {}).values())

    onsets = stage("onsets", lambda: backfill_onsets(limit=40))
    res.onsets_fixed = (onsets or {}).get("fixed", 0)

    clusters = stage("clusters", lambda: build_all())
    res.clusters_built = (clusters or {}).get("with_pre_move", 0)

    if not skip_attribution:
        res.attributed = stage("attribute",
                               lambda: _attribute(attribution_limit, attribute_since)) or {}

    logger.info("batch complete: %s", res.summary())
    return res


def _attribute(limit: int, since: date | None = None) -> dict[str, int]:
    """Attribute unattributed swings, sectors first, newest first."""
    from swing.agent.graph import attribute_swing
    from swing.store.session import connect

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.ticker, s.d, s.residual_z
            FROM swings s
            WHERE NOT EXISTS (SELECT 1 FROM attributions a
                              WHERE a.swing_id=s.id AND a.run_kind='production')
              AND s.onset_ts IS NOT NULL
              AND EXISTS (SELECT 1 FROM clusters c
                          WHERE c.swing_id=s.id AND c.timing='pre_move')
              AND (%(since)s::date IS NULL OR s.d >= %(since)s::date)
            ORDER BY (s.entity_type='sector') DESC, s.d DESC, abs(s.residual_z) DESC
            LIMIT %(limit)s
            """,
            {"since": since, "limit": limit},
        ).fetchall()

    counts: dict[str, int] = {}
    for r in rows:
        try:
            out = attribute_swing(r["id"])
        except Exception as exc:  # noqa: BLE001 - quota exhaustion ends the stage cleanly
            logger.warning("attribution stopped at %s %s: %s",
                           r["ticker"], r["d"], str(exc)[:90])
            counts["stopped"] = counts.get("stopped", 0) + 1
            break
        if out.get("verdict_reason") == "llm_error":
            logger.warning("attribution stopped at %s %s: model call failed",
                           r["ticker"], r["d"])
            counts["stopped"] = counts.get("stopped", 0) + 1
            break
        attr = out.get("attribution")
        v = attr.verdict if attr else "error"
        counts[v] = counts.get(v, 0) + 1
    return counts
