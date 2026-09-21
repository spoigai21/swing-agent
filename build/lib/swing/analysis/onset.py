"""Locate WHEN a move actually happened.

⚠️ This is the plan's second irrecoverable mistake, and the only bug in it that
passes every gate while producing wrong answers. agent-plan.md 1.3.

If NVDA closes down 3.5%, the drop may have happened at 09:35 with the stock
flat afterwards. A 14:00 article is then POST-move commentary, but naive code
sees "published before the closing bar" and files it as a candidate catalyst —
rebuilding the reactive-journalism trap by accident, invisibly.

So T = onset_ts, never the closing timestamp. Every window in Phase 2 keys off
the value this module returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time

import numpy as np

from swing.common import logging as log
from swing.common.timeutil import assert_utc
from swing.ingest.config import thresholds
from swing.store.session import connect

logger = log.get("analysis.onset")


@dataclass(slots=True)
class Onset:
    ts: datetime | None
    swing_type: str          # gap | intraday | mixed | unknown
    source: str              # intraday | fallback_48h
    gap_share: float | None = None


def _intraday(ticker: str, d: date) -> tuple[list[datetime], np.ndarray]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT ts, close FROM intraday_bars "
            "WHERE ticker=%s AND ts::date=%s AND interval_sec=300 ORDER BY ts",
            (ticker, d),
        ).fetchall()
    return [r["ts"] for r in rows], np.array([float(r["close"]) for r in rows])


def _prev_close(ticker: str, d: date) -> float | None:
    with connect() as conn:
        r = conn.execute(
            "SELECT close FROM bars WHERE ticker=%s AND ts::date < %s ORDER BY ts DESC LIMIT 1",
            (ticker, d),
        ).fetchone()
    return float(r["close"]) if r else None


def locate(ticker: str, d: date, sector_etf: str | None, market_etf: str,
           beta_mkt: float, beta_sector: float | None) -> Onset:
    """Find the start of the largest sustained abnormal-return run.

    Deviation from agent-plan.md 1.3's sketch: that neutralises only the sector
    intraday, while the daily residual this is locating is net of BOTH betas.
    Using both keeps the intraday quantity the same one we detected on.
    """
    cfg = thresholds()["onset"]
    ts, px = _intraday(ticker, d)
    prev = _prev_close(ticker, d)
    if len(px) < 4 or prev is None:
        return fallback(d)

    r = np.diff(np.log(px), prepend=np.log(prev))

    # Neutralise both factors bar-by-bar, aligned on timestamp.
    ar = r.copy()
    for etf, beta in ((market_etf, beta_mkt), (sector_etf, beta_sector)):
        if not etf or beta is None:
            continue
        f_ts, f_px = _intraday(etf, d)
        f_prev = _prev_close(etf, d)
        if len(f_px) < 4 or f_prev is None:
            logger.debug("%s %s: no intraday for factor %s", ticker, d, etf)
            continue
        f_r = np.diff(np.log(f_px), prepend=np.log(f_prev))
        aligned = dict(zip(f_ts, f_r, strict=True))
        ar = ar - np.array([beta * aligned.get(t, 0.0) for t in ts])

    car = np.cumsum(ar)
    gap_ar = float(ar[0])                       # overnight, prev close -> open
    total = float(car[-1])
    if total == 0:
        return fallback(d)

    gap_share = abs(gap_ar) / abs(total)
    if gap_share > float(cfg["gap_dominant_ratio"]):
        return Onset(assert_utc(ts[0]), "gap", "intraday", gap_share)
    if gap_share > float(cfg["gap_mixed_ratio"]):
        return Onset(assert_utc(ts[0]), "mixed", "intraday", gap_share)

    # Intraday: the run starts at the extreme BEFORE the move, in the direction
    # opposite the move itself.
    direction = np.sign(total - car[0])
    onset_idx = int(np.argmin(direction * car))
    return Onset(assert_utc(ts[onset_idx]), "intraday", "intraday", gap_share)


def fallback(d: date) -> Onset:
    """No intraday bars. Conservative 48h window, flagged as weaker evidence.

    These records must be excluded from ranking-weight tuning. With Tiingo
    supplying 5-minute bars back to at least 2024-09-03, this should now be
    rare — it applies only when a fetch genuinely fails.
    """
    # Anchor at the closing bell (20:00 UTC / 16:00 ET). The window WIDTH is
    # applied in windows.py, which owns that decision for every swing_type.
    close_utc = datetime.combine(d, time(20, 0), tzinfo=UTC)
    return Onset(close_utc, "unknown", "fallback_48h", None)
