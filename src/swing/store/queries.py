"""Every read the rest of the system needs, in one place and typed.

Nothing outside this module writes SQL against the analysis tables.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from swing.store.session import connect

# --------------------------- articles -------------------------------------

def article_counts_by_source() -> list[dict[str, Any]]:
    with connect() as conn:
        return conn.execute(
            """
            SELECT source,
                   count(*) FILTER (WHERE published_at > now() - interval '24 hours') AS last_24h,
                   count(*) FILTER (WHERE published_at > now() - interval '7 days')   AS last_7d,
                   count(*) AS total,
                   max(retrieved_at) AS last_retrieved
            FROM articles_raw GROUP BY source ORDER BY total DESC
            """
        ).fetchall()


def unnormalized_batch(limit: int = 256) -> list[dict[str, Any]]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM articles_raw WHERE normalized = false ORDER BY id LIMIT %s",
            (limit,),
        ).fetchall()


# --------------------------- prices ---------------------------------------

def bars_for(ticker: str, start: date | None = None, end: date | None = None) -> list[dict]:
    sql = "SELECT ticker, ts, open, high, low, close, volume FROM bars WHERE ticker = %s"
    params: list[Any] = [ticker]
    if start:
        sql += " AND ts >= %s"
        params.append(start)
    if end:
        sql += " AND ts <= %s"
        params.append(end)
    sql += " ORDER BY ts"
    with connect() as conn:
        return conn.execute(sql, tuple(params)).fetchall()


def bar_coverage() -> list[dict[str, Any]]:
    """Per-ticker span and count. The Gate 0 check for missing trading days."""
    with connect() as conn:
        return conn.execute(
            """
            SELECT ticker, count(*) AS n,
                   min(ts)::date AS first_day, max(ts)::date AS last_day
            FROM bars GROUP BY ticker ORDER BY ticker
            """
        ).fetchall()


def intraday_for(ticker: str, d: date, interval_sec: int = 300) -> list[dict]:
    with connect() as conn:
        return conn.execute(
            """
            SELECT ts, open, high, low, close, volume FROM intraday_bars
            WHERE ticker = %s AND interval_sec = %s AND ts::date = %s ORDER BY ts
            """,
            (ticker, interval_sec, d),
        ).fetchall()


# --------------------------- analysis -------------------------------------

def factors_for(ticker: str, days: int = 90) -> list[dict[str, Any]]:
    with connect() as conn:
        return conn.execute(
            """
            SELECT * FROM daily_factors
            WHERE ticker = %s AND d > current_date - %s::int
            ORDER BY d DESC
            """,
            (ticker, days),
        ).fetchall()


def idio_share_summary(tickers: list[str], days: int = 90) -> list[dict[str, Any]]:
    """Backs `swing compare`.

    idio_share here is VARIANCE-based: var(residual) / var(ret) over the window.
    The per-day ratio |residual|/|ret| stored on daily_factors is far worse —
    averaging it gives values like 3.29 because days with near-zero `ret`
    dominate.

    ⚠️ This is NOT bounded by 1. Betas are fitted on a trailing 120-day window
    and applied to the following day, so the residual is OUT-OF-SAMPLE; only
    in-sample OLS guarantees var(residual) <= var(y). SBUX comes in at 1.11 with
    R² = 0.14, meaning the factor adjustment ADDS variance for that name rather
    than removing it.

    That is a diagnostic, not a defect: a ratio above 1 says the assigned factor
    is a poor fit and the "residual" is mostly noise the model introduced. It is
    the same check data-sources.md A.2 asks for on MU/SMH, generalised.
    """
    with connect() as conn:
        return conn.execute(
            """
            SELECT ticker,
                   count(*) AS n,
                   round((var_samp(residual) / nullif(var_samp(ret), 0))::numeric, 3)
                                                              AS avg_idio_share,
                   round(avg(r_squared)::numeric, 3)           AS avg_r2,
                   round(stddev_samp(residual)::numeric, 5)    AS residual_vol,
                   count(*) FILTER (WHERE abs(residual_z) >= 2) AS swing_days
            FROM daily_factors
            WHERE ticker = ANY(%s) AND d > current_date - %s::int AND status = 'ok'
            GROUP BY ticker ORDER BY avg_idio_share DESC NULLS LAST
            """,
            (tickers, days),
        ).fetchall()


def swings_for(ticker: str | None = None, days: int = 90) -> list[dict[str, Any]]:
    sql = """
        SELECT s.*, a.verdict, a.unexplained_note
        FROM swings s
        LEFT JOIN LATERAL (
            SELECT verdict, unexplained_note FROM attributions
            WHERE swing_id = s.id ORDER BY created_at DESC LIMIT 1
        ) a ON true
        WHERE s.d > current_date - %s::int AND s.superseded_by IS NULL
    """
    params: list[Any] = [days]
    if ticker:
        sql += " AND s.ticker = %s"
        params.append(ticker)
    sql += " ORDER BY s.d DESC, abs(s.residual_z) DESC"
    with connect() as conn:
        return conn.execute(sql, tuple(params)).fetchall()


def latest_swing(ticker: str) -> dict[str, Any] | None:
    with connect() as conn:
        return conn.execute(
            """
            SELECT * FROM swings WHERE ticker = %s AND superseded_by IS NULL
            ORDER BY d DESC LIMIT 1
            """,
            (ticker,),
        ).fetchone()


def unexplained_rate_by_ticker(days: int = 90) -> list[dict[str, Any]]:
    """The single most useful number in the dashboard: a high rate is a SOURCE
    GAP, not a model problem. agent-plan.md 6.4."""
    with connect() as conn:
        return conn.execute(
            """
            SELECT s.ticker,
                   count(*) AS n,
                   round((count(*) FILTER (WHERE a.verdict = 'unexplained'))::numeric
                         / nullif(count(*), 0), 3) AS unexplained_rate
            FROM attributions a JOIN swings s ON s.id = a.swing_id
            WHERE a.created_at > now() - (%s::int || ' days')::interval
              AND a.run_kind = 'production'
            GROUP BY s.ticker ORDER BY unexplained_rate DESC NULLS LAST
            """,
            (days,),
        ).fetchall()
