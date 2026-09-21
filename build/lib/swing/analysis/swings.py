"""Swing detection on the RESIDUAL, plus drift, volume context and earnings mode.

Threshold the residual normalised by its own trailing volatility, never a fixed
percentage: 4% is an earthquake for a utility and an ordinary day for a small-cap
biotech. agent-plan.md 1.2.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta

import numpy as np

from swing.analysis.factors import Entity, entities
from swing.analysis.onset import fallback, locate
from swing.common import logging as log
from swing.common.timeutil import UTC
from swing.ingest.config import thresholds
from swing.ingest.prices import fetch_intraday
from swing.store.session import connect

logger = log.get("analysis.swings")


@dataclass(slots=True)
class Swing:
    ticker: str
    d: date
    kind: str                  # daily | drift
    drift_window: int | None
    residual: float
    residual_z: float
    total_return: float
    market_component: float
    sector_component: float
    volume_z: float | None
    swing_type: str
    onset_ts: datetime | None
    onset_source: str | None
    earnings_mode: bool
    entity_type: str


def _factors(ticker: str) -> list[dict]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM daily_factors WHERE ticker=%s AND status='ok' ORDER BY d",
            (ticker,),
        ).fetchall()


def _earnings_days(ticker: str) -> set[date]:
    """Days with an 8-K Item 2.02 (results of operations).

    agent-plan.md 1.7: do NOT suppress earnings. They are the largest driver of
    idiosyncratic moves, so going dark on them means going dark on the four days
    a year per ticker that matter most. Flag them instead so the agent's job
    changes from 'find the catalyst' to 'characterise it'.
    """
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT published_at::date AS d FROM articles_raw
            WHERE raw->>'ticker' = %s AND raw->>'form' LIKE '8-K%%'
              AND raw->>'items' LIKE %s
            """,
            (ticker, "%2.02%"),
        ).fetchall()
    # An 8-K filed after the close drives the NEXT session, so count both.
    days = set()
    for r in rows:
        days.add(r["d"])
        days.add(r["d"] + timedelta(days=1))
    return days


def _daily_swing(entity: Entity, row: dict, onset, earnings_mode: bool) -> Swing:
    return Swing(
        ticker=entity.ticker, d=row["d"], kind="daily", drift_window=None,
        residual=float(row["residual"]), residual_z=float(row["residual_z"]),
        total_return=float(row["ret"]),
        market_component=float(row["market_component"]),
        sector_component=float(row["sector_component"]),
        volume_z=float(row["volume_z"]) if row["volume_z"] is not None else None,
        swing_type=onset.swing_type, onset_ts=onset.ts,
        onset_source=onset.source, earnings_mode=earnings_mode,
        entity_type=entity.entity_type,
    )


def detect_day(entity: Entity, d: date) -> int | None:
    """Detect, time and store the daily swing for ONE day. Returns its id, or
    None when the day is not a swing.

    A question needs one day, not a history scan. Intraday bars for the stock
    and its factor ETFs are captured first, so the onset is exact rather than
    the 48h fallback.
    """
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM daily_factors WHERE ticker=%s AND d=%s AND status='ok'",
            (entity.ticker, d)).fetchone()
    if not row or abs(float(row["residual_z"])) < float(thresholds()["swings"]["z_threshold"]):
        return None
    onset = _onset_for(entity, d, row, capture=True)
    earnings = entity.entity_type == "stock" and d in _earnings_days(entity.ticker)
    with connect() as conn:
        return conn.execute(UPSERT_DAILY + "RETURNING id",
                            asdict(_daily_swing(entity, row, onset, earnings))).fetchone()["id"]


def detect(entity: Entity, capture_intraday: bool = True) -> list[Swing]:
    cfg = thresholds()["swings"]
    z_thr = float(cfg["z_threshold"])
    drift_thr = float(cfg["drift_z_threshold"])
    rows = _factors(entity.ticker)
    if not rows:
        return []

    earnings = _earnings_days(entity.ticker) if entity.entity_type == "stock" else set()
    resid = np.array([float(r["residual"]) for r in rows])
    out: list[Swing] = []
    daily_days: set[date] = set()

    for r in rows:
        if abs(float(r["residual_z"])) < z_thr:
            continue
        daily_days.add(r["d"])
        onset = _onset_for(entity, r["d"], r, capture_intraday)
        out.append(_daily_swing(entity, r, onset, r["d"] in earnings))

    # Multi-day drift: a stock bleeding 1.3 sigma a day for four days never
    # fires a single-day threshold, and that is often the better story.
    for w in [int(x) for x in cfg["drift_windows"]]:
        for i in range(w, len(rows)):
            car = float(resid[i - w + 1:i + 1].sum())
            vol = float(rows[i]["residual_vol_60"] or 0)
            if vol <= 0:
                continue
            cz = car / (vol * np.sqrt(w))
            if abs(cz) < drift_thr:
                continue
            d = rows[i]["d"]
            # Deduplicate against single-day swings: one big day inside a 5-day
            # window is the same event, not two.
            span = {rows[j]["d"] for j in range(i - w + 1, i + 1)}
            if span & daily_days:
                continue
            out.append(Swing(
                ticker=entity.ticker, d=d, kind="drift", drift_window=w,
                residual=car, residual_z=float(cz),
                total_return=float(rows[i]["ret"]),
                market_component=float(rows[i]["market_component"]),
                sector_component=float(rows[i]["sector_component"]),
                volume_z=float(rows[i]["volume_z"]) if rows[i]["volume_z"] is not None else None,
                swing_type="drift",
                onset_ts=datetime.combine(rows[i - w + 1]["d"], time(13, 30), tzinfo=UTC),
                onset_source=None, earnings_mode=d in earnings,
                entity_type=entity.entity_type,
            ))
    return out


def has_intraday(ticker: str, d: date) -> bool:
    with connect() as conn:
        return bool(conn.execute(
            "SELECT 1 FROM intraday_bars WHERE ticker=%s AND ts::date=%s LIMIT 1", (ticker, d)
        ).fetchone())


def capture_intraday_for(entity: Entity, d: date) -> int:
    """Fetch intraday for the ticker and its factors, skipping what we hold.

    Capture-on-detect is what makes the intraday lookback limit a backfill
    constraint rather than an operational one. Skipping cached symbols matters:
    the market and sector ETFs repeat across every ticker on the same date, so
    without this the same SPY day is refetched a dozen times.
    """
    got = 0
    for sym in filter(None, [entity.ticker, entity.market_etf, entity.sector_etf]):
        if has_intraday(sym, d):
            continue
        try:
            got += fetch_intraday(sym, d)
        except Exception:
            logger.exception("intraday capture failed for %s %s", sym, d)
    return got


def _onset_for(entity: Entity, d: date, row: dict, capture: bool):
    if capture:
        capture_intraday_for(entity, d)
    try:
        return locate(entity.ticker, d, entity.sector_etf, entity.market_etf,
                      float(row["beta_mkt"]),
                      float(row["beta_sector"]) if row["beta_sector"] is not None else None)
    except Exception:
        logger.exception("onset failed for %s %s", entity.ticker, d)
        return fallback(d)


_UPSERT_COLS = """
INSERT INTO swings (ticker, d, kind, drift_window, residual, residual_z, total_return,
                    market_component, sector_component, volume_z, swing_type,
                    onset_ts, onset_source, earnings_mode, entity_type)
VALUES (%(ticker)s, %(d)s, %(kind)s, %(drift_window)s, %(residual)s, %(residual_z)s,
        %(total_return)s, %(market_component)s, %(sector_component)s, %(volume_z)s,
        %(swing_type)s, %(onset_ts)s, %(onset_source)s, %(earnings_mode)s, %(entity_type)s)
"""

_UPSERT_SET = """
DO UPDATE SET
  residual=EXCLUDED.residual, residual_z=EXCLUDED.residual_z,
  total_return=EXCLUDED.total_return, market_component=EXCLUDED.market_component,
  sector_component=EXCLUDED.sector_component, volume_z=EXCLUDED.volume_z,
  swing_type=EXCLUDED.swing_type, onset_ts=EXCLUDED.onset_ts,
  onset_source=EXCLUDED.onset_source, earnings_mode=EXCLUDED.earnings_mode
"""

# Two statements because the uniqueness is enforced by two PARTIAL indexes, and
# ON CONFLICT must name the index predicate to use one.
UPSERT_DAILY = _UPSERT_COLS + "ON CONFLICT (ticker, d, kind) WHERE drift_window IS NULL " + _UPSERT_SET
UPSERT_DRIFT = (_UPSERT_COLS
                + "ON CONFLICT (ticker, d, kind, drift_window) WHERE drift_window IS NOT NULL "
                + _UPSERT_SET)


def backfill_onsets(limit: int | None = None) -> dict[str, int]:
    """Resumable: capture intraday and recompute onset for swings still on the
    fallback. Separate from detect_all() because a full historical backfill is
    hundreds of API calls and must be restartable."""
    from swing.analysis.factors import entities as _entities

    by_ticker = {e.ticker: e for e in _entities()}
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, ticker, d FROM swings WHERE onset_source IS DISTINCT FROM 'intraday' "
            "AND kind='daily' ORDER BY d DESC" + (f" LIMIT {int(limit)}" if limit else "")
        ).fetchall()
    fixed, failed = 0, 0
    for r in rows:
        e = by_ticker.get(r["ticker"])
        if not e:
            continue
        capture_intraday_for(e, r["d"])
        with connect() as conn:
            f = conn.execute(
                "SELECT beta_mkt, beta_sector FROM daily_factors WHERE ticker=%s AND d=%s",
                (r["ticker"], r["d"])).fetchone()
        if not f:
            failed += 1
            continue
        o = _onset_for(e, r["d"], f, capture=False)
        if o.source != "intraday":
            failed += 1
            continue
        with connect() as conn:
            conn.execute(
                "UPDATE swings SET onset_ts=%s, swing_type=%s, onset_source=%s WHERE id=%s",
                (o.ts, o.swing_type, o.source, r["id"]))
        fixed += 1
    logger.info("onset backfill: %d fixed, %d still without intraday", fixed, failed)
    return {"fixed": fixed, "failed": failed, "considered": len(rows)}


def detect_all(capture_intraday: bool = False, since: date | None = None) -> dict[str, int]:
    out = {}
    for e in entities():
        swings = detect(e, capture_intraday=capture_intraday)
        if since:
            swings = [s for s in swings if s.d >= since]
        if swings:
            daily = [asdict(s) for s in swings if s.drift_window is None]
            drift = [asdict(s) for s in swings if s.drift_window is not None]
            with connect() as conn, conn.cursor() as cur:
                if daily:
                    cur.executemany(UPSERT_DAILY, daily)
                if drift:
                    cur.executemany(UPSERT_DRIFT, drift)
        out[e.ticker] = len(swings)
        logger.info("%s: %d swings", e.ticker, len(swings))
    return out


def unexplained_note(volume_z: float | None) -> str:
    """Volume turns a dead-end verdict into a usable signal. agent-plan.md 3.2."""
    base = "No pre-move catalyst identified from credible sources. "
    cfg = thresholds()["swings"]
    if volume_z is None:
        return base + "Volume context unavailable."
    if volume_z >= float(cfg["volume_flow_z"]):
        return base + (
            f"Volume was {volume_z:.1f}σ above normal, consistent with a flow event "
            "— index activity, a block trade, or a position unwind — rather than news.")
    if volume_z <= float(cfg["volume_thin_z"]):
        return base + ("Volume was below normal, so the move may reflect thin liquidity "
                       "rather than a catalyst.")
    return base + ("Volume was unremarkable. The move may reflect positioning or "
                   "information not captured in the monitored sources.")
