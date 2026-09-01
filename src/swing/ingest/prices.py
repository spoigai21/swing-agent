"""Price ingestion. yfinance is the source for ALL bars, daily and intraday.

agent-plan.md 0.3 says "yfinance for the historical backfill and Finnhub for
ongoing updates". That does not work: Finnhub's free tier returns 403 on
/stock/candle for both intraday and daily resolutions (verified 2026-08-31), and
/quote carries no volume. Tiingo remains the documented fallback.

UNADJUSTED OHLCV only. Adjusted series get silently rewritten on every split or
dividend, so a backtest changes results month to month for no visible reason.
Splits and dividends go to corporate_actions and are applied at query time.
agent-plan.md 0.2.
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd

from swing.common import logging as log
from swing.common.timeutil import assert_utc
from swing.ingest.config import all_symbols, stocks
from swing.store.session import connect

logger = log.get("ingest.prices")

UPSERT_BAR = """
INSERT INTO bars (ticker, ts, open, high, low, close, volume, source)
VALUES (%(ticker)s, %(ts)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(source)s)
ON CONFLICT (ticker, ts) DO UPDATE SET
  open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
  close = EXCLUDED.close, volume = EXCLUDED.volume, source = EXCLUDED.source
"""

UPSERT_INTRADAY = """
INSERT INTO intraday_bars (ticker, ts, interval_sec, open, high, low, close, volume)
VALUES (%(ticker)s, %(ts)s, %(interval_sec)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s)
ON CONFLICT (ticker, ts, interval_sec) DO UPDATE SET
  open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
  close = EXCLUDED.close, volume = EXCLUDED.volume
"""

UPSERT_ACTION = """
INSERT INTO corporate_actions (ticker, ex_date, kind, ratio, amount)
VALUES (%(ticker)s, %(ex_date)s, %(kind)s, %(ratio)s, %(amount)s)
ON CONFLICT (ticker, ex_date, kind) DO UPDATE SET
  ratio = EXCLUDED.ratio, amount = EXCLUDED.amount
"""


def _flatten(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """yfinance returns a MultiIndex column frame when given a list of tickers,
    and a flat one for a single ticker. Normalise to flat."""
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(symbol, axis=1, level=1)
        except KeyError:
            df = df.droplevel(1, axis=1)
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return df


def _rows(df: pd.DataFrame, ticker: str, source: str) -> list[dict]:
    out = []
    for ts, r in df.iterrows():
        ts = pd.Timestamp(ts)
        # Daily bars come back tz-naive; stamp them UTC rather than let a naive
        # datetime reach the database.
        ts = ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)
        if pd.isna(r.get("close")):
            continue
        out.append({
            "ticker": ticker,
            "ts": assert_utc(ts.to_pydatetime()),
            "open": _num(r.get("open")), "high": _num(r.get("high")),
            "low": _num(r.get("low")), "close": _num(r.get("close")),
            "volume": int(r["volume"]) if not pd.isna(r.get("volume")) else None,
            "source": source,
        })
    return out


def _num(v) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


def backfill_daily(symbols: list[str] | None = None, years: int = 2) -> dict[str, int]:
    """Pull `years` of unadjusted daily OHLCV for every symbol."""
    import yfinance as yf

    symbols = symbols or all_symbols()
    start = (datetime.now(UTC).date() - timedelta(days=int(365.25 * years) + 5)).isoformat()
    written: dict[str, int] = {}

    for sym in symbols:
        try:
            df = yf.download(sym, start=start, interval="1d", progress=False,
                             auto_adjust=False, actions=True)
        except Exception:
            logger.exception("download failed for %s", sym)
            written[sym] = 0
            continue
        if df is None or df.empty:
            logger.warning("%s: no data returned", sym)
            written[sym] = 0
            continue

        df = _flatten(df, sym)
        rows = _rows(df, sym, "yfinance")
        actions = _action_rows(df, sym)
        with connect() as conn, conn.cursor() as cur:
            cur.executemany(UPSERT_BAR, rows)
            if actions:
                cur.executemany(UPSERT_ACTION, actions)
        written[sym] = len(rows)
        logger.info("%s: %d daily bars, %d corporate actions", sym, len(rows), len(actions))
    return written


def _action_rows(df: pd.DataFrame, ticker: str) -> list[dict]:
    out = []
    for ts, r in df.iterrows():
        ex = pd.Timestamp(ts).date()
        split = r.get("stock_splits")
        div = r.get("dividends")
        if split and not pd.isna(split) and float(split) != 0.0:
            out.append({"ticker": ticker, "ex_date": ex, "kind": "split",
                        "ratio": float(split), "amount": None})
        if div and not pd.isna(div) and float(div) != 0.0:
            out.append({"ticker": ticker, "ex_date": ex, "kind": "dividend",
                        "ratio": None, "amount": float(div)})
    return out


def fetch_intraday(ticker: str, day: date, interval_sec: int = 300) -> int:
    """Capture one day of 5-minute bars. Call this the MOMENT a swing is detected.

    yfinance serves 5m bars for the last 60 days only (verified: period=2mo is
    rejected). Capture-on-detect means that cap limits backfill but never
    forward operation. agent-plan.md 1.3.
    """
    import yfinance as yf

    interval = f"{interval_sec // 60}m"
    start = day.isoformat()
    end = (day + timedelta(days=1)).isoformat()
    df = yf.download(ticker, start=start, end=end, interval=interval,
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        logger.warning("%s %s: no %s bars (older than the 60-day window?)",
                       ticker, day, interval)
        return 0
    df = _flatten(df, ticker)
    rows = [{**r, "interval_sec": interval_sec} for r in _rows(df, ticker, "yfinance")]
    for r in rows:
        r.pop("source", None)
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(UPSERT_INTRADAY, rows)
    logger.info("%s %s: %d intraday bars", ticker, day, len(rows))
    return len(rows)


def coverage_report() -> list[dict]:
    """Per-symbol span, count, and gap check against the NYSE calendar proxy."""
    with connect() as conn:
        return conn.execute(
            """
            SELECT ticker, count(*) AS n,
                   min(ts)::date AS first_day, max(ts)::date AS last_day
            FROM bars GROUP BY ticker ORDER BY ticker
            """
        ).fetchall()


def enforce_history_start(purge: bool = False) -> list[tuple[str, bool, str]]:
    """Drop bars predating a ticker's `history_starts` date.

    Two distinct things produce early bars, and they need different responses:

    * **Spliced predecessor** — a source joins a same-ticker predecessor company.
      For SNDK that would be the pre-2016 SanDisk acquired by Western Digital.
      A beta fitted across that join spans two unrelated businesses.
    * **When-issued trading** — before a spin-off completes, the new entity
      trades on a when-issued basis for a week or two. Same company, but a
      different instrument, on ~24x thinner volume with erratic pricing.
      Verified for SNDK: 6 bars from 2025-02-13, median volume 406,800 against
      9,678,900 after regular-way trading opened on 2025-02-24.

    Both corrupt beta estimation, so both are removed. data-sources.md A.2.
    """
    results = []
    for ticker, meta in stocks().items():
        expected = meta.get("history_starts")
        if not expected:
            continue
        with connect() as conn:
            row = conn.execute(
                "SELECT min(ts)::date AS first_day, count(*) n FROM bars WHERE ticker=%s",
                (ticker,),
            ).fetchone()
            if not row or row["n"] == 0:
                results.append((ticker, False, "no bars"))
                continue
            first = row["first_day"]
            if first.isoformat() >= expected:
                results.append((ticker, True, f"starts {first} ({row['n']} bars), as expected"))
                continue

            early = conn.execute(
                "SELECT count(*) n FROM bars WHERE ticker=%s AND ts::date < %s",
                (ticker, expected),
            ).fetchone()["n"]
            gap_days = (date.fromisoformat(expected) - first).days
            cause = ("spliced predecessor company" if gap_days > 365
                     else "when-issued pre-spin-off trading")
            if purge:
                conn.execute(
                    "DELETE FROM bars WHERE ticker=%s AND ts::date < %s", (ticker, expected)
                )
                results.append((ticker, True,
                                f"purged {early} bars before {expected} ({cause})"))
            else:
                results.append((
                    ticker, False,
                    f"{early} bars before {expected} - {cause}. Re-run with purge=True.",
                ))
    return results
