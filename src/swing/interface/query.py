"""Natural-language query over STORED attributions. agent-plan.md 6.2.

Two properties worth stating:

* **The forecast guardrail runs first**, before any model call.
* **Most queries never reach the model.** Recognised intents are answered by
  SQL, which is faster, free, and cannot hallucinate. The LLM is a fallback for
  phrasing the deterministic result, never a source of facts.

⚠️ No conversation memory, deliberately. `swing-cli.md` Part 6: carrying context
between turns lets a small-sample caveat from one question silently shape the
answer to another, and you cannot see it happen.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from swing.interface.guardrail import check as guard_check


@dataclass(slots=True)
class Answer:
    text: str
    used_llm: bool = False
    rows: list | None = None


_TICKER = re.compile(r"\b([A-Z]{1,5})\b")


def _tickers_in(q: str) -> list[str]:
    from swing.ingest.config import sectors, stocks

    known = set(stocks()) | set(sectors())
    return [t for t in _TICKER.findall(q) if t in known]


def _days_in(q: str, default: int = 30) -> int:
    m = re.search(r"\b(?:last|past)\s+(\d+)\s*(day|week|month|year)", q, re.IGNORECASE)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        return n * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
    if re.search(r"\b(this|last)\s+week\b", q, re.IGNORECASE):
        return 7
    if re.search(r"\b(this|last)\s+month\b", q, re.IGNORECASE):
        return 30
    if re.search(r"\b(this|last)\s+year\b", q, re.IGNORECASE):
        return 365
    return default


def _date_in(q: str) -> date | None:
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", q)
    return date.fromisoformat(m.group(1)) if m else None


def answer(question: str, allow_llm: bool = True) -> Answer:
    """Answer a question, refusing forward-looking ones before any model call."""
    g = guard_check(question)
    if not g.allowed:
        return Answer(g.reason or "refused", used_llm=False)

    q = question.strip()
    tickers = _tickers_in(q)
    low = q.lower()

    if re.search(r"\bunexplained\b", low):
        return _unexplained(tickers[0] if tickers else None, _days_in(q))
    if re.search(r"\b(idio|idiosyncratic|company[- ]specific)\b", low):
        return _idio(tickers, _days_in(q, 90))
    if re.search(r"\b(why|what (drove|caused|moved)|explain)\b", low) and tickers:
        return _why(tickers[0], _date_in(q))
    if re.search(r"\b(biggest|largest|top)\b.*\b(move|swing)\b", low):
        return _biggest(tickers[0] if tickers else None, _days_in(q))
    if re.search(r"\b(coverage|how many articles|what data)\b", low):
        return _coverage()

    if allow_llm:
        return _llm_fallback(q)
    return Answer(
        "I could not map that to a known query. Try: 'why did NVDA move on "
        "2026-08-27', 'show unexplained swings this month', or "
        "'which tickers have the highest idiosyncratic share'.")


def _why(ticker: str, d: date | None) -> Answer:
    from swing.analysis.decompose import load as load_decomp
    from swing.store.session import connect

    with connect() as conn:
        if d:
            sw = conn.execute(
                "SELECT * FROM swings WHERE ticker=%s AND d=%s AND kind='daily'",
                (ticker, d)).fetchone()
        else:
            sw = conn.execute(
                "SELECT * FROM swings WHERE ticker=%s AND kind='daily' "
                "ORDER BY d DESC LIMIT 1", (ticker,)).fetchone()
        if not sw:
            return Answer(f"No swing on record for {ticker}"
                          + (f" on {d}." if d else "."))
        att = conn.execute(
            "SELECT payload, verdict, unexplained_note FROM attributions "
            "WHERE swing_id=%s AND run_kind='production' "
            "ORDER BY created_at DESC LIMIT 1", (sw["id"],)).fetchone()

    dec = load_decomp(ticker, sw["d"])
    lines = [dec.sentence() if dec else f"{ticker} {sw['d']}"]
    if not att:
        lines.append("\nNo attribution stored yet. Run: "
                     f"swing why {ticker} --date {sw['d']}")
        return Answer("\n".join(lines), rows=[dict(sw)])

    lines.append(f"\nVerdict: {att['verdict']}")
    for c in (att["payload"] or {}).get("candidates") or []:
        lines.append(f"  [{c['confidence']}] {c['event_type']}: {c['catalyst']}")
        for e in c.get("evidence") or []:
            lines.append(f"    - ({e['timing']}, tier {e['source_tier']}) "
                         f"{e['headline'][:64]}")
    if att["unexplained_note"]:
        lines.append(f"\n{att['unexplained_note']}")
    return Answer("\n".join(lines))


def _unexplained(ticker: str | None, days: int) -> Answer:
    from swing.store.queries import LATEST_PRODUCTION
    from swing.store.session import connect

    sql = """
        SELECT s.ticker, s.d, s.residual_z, s.volume_z, a.unexplained_note
        FROM attributions a JOIN swings s ON s.id=a.swing_id
        WHERE a.verdict='unexplained' AND a.run_kind='production'
          AND s.d > current_date - %s::int
          AND """ + LATEST_PRODUCTION + """
    """
    params: list = [days]
    if ticker:
        sql += " AND s.ticker=%s"
        params.append(ticker)
    sql += " ORDER BY abs(s.residual_z) DESC LIMIT 25"
    with connect() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    if not rows:
        return Answer(f"No unexplained swings in the last {days} days"
                      + (f" for {ticker}." if ticker else "."))
    lines = [f"{len(rows)} unexplained swing(s) in the last {days} days.",
             "These mark gaps in source coverage and are worth reviewing:\n"]
    for r in rows:
        lines.append(f"  {r['ticker']:<6} {r['d']}  z={float(r['residual_z']):+.2f}  "
                     f"vol_z={float(r['volume_z'] or 0):+.1f}")
    return Answer("\n".join(lines), rows=[dict(r) for r in rows])


def _idio(tickers: list[str], days: int) -> Answer:
    from swing.ingest.config import stocks
    from swing.store import queries

    rows = queries.idio_share_summary(tickers or list(stocks()), days)
    if not rows:
        return Answer("No factor rows for that window.")
    header = (f"Idiosyncratic share over the last {days} days "
              "(var(residual)/var(return)):\n")
    lines = [header]
    flagged = False
    for r in rows:
        share = float(r["avg_idio_share"] or 0)
        mark = ""
        if share > 1.0:
            mark = "  ⚠ factor model adds variance"
            flagged = True
        lines.append(f"  {r['ticker']:<6} {share:>5.2f}   "
                     f"R²={float(r['avg_r2'] or 0):.2f}  swings={r['swing_days']}{mark}")
    if flagged:
        lines.append(
            "\n  ⚠ Values above 1.0 are possible because betas are fitted "
            "out-of-sample.\n    They mean the assigned sector factor is a poor "
            "fit for that name and the\n    residual is partly noise the model "
            "introduced — worth revisiting the factor.")
    return Answer("\n".join(lines), rows=[dict(r) for r in rows])


def _biggest(ticker: str | None, days: int) -> Answer:
    from swing.store.session import connect

    sql = ("SELECT ticker, d, residual_z, total_return, swing_type, earnings_mode "
           "FROM swings WHERE kind='daily' AND d > current_date - %s::int")
    params: list = [days]
    if ticker:
        sql += " AND ticker=%s"
        params.append(ticker)
    sql += " ORDER BY abs(residual_z) DESC LIMIT 10"
    with connect() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    if not rows:
        return Answer(f"No swings in the last {days} days.")
    lines = [f"Largest idiosyncratic moves, last {days} days:\n"]
    for r in rows:
        lines.append(f"  {r['ticker']:<6} {r['d']}  z={float(r['residual_z']):+6.2f}  "
                     f"return={float(r['total_return'])*100:+6.2f}%  {r['swing_type']}"
                     + ("  [earnings]" if r["earnings_mode"] else ""))
    return Answer("\n".join(lines), rows=[dict(r) for r in rows])


def _coverage() -> Answer:
    from swing.commands import coverage_data

    d = coverage_data()
    return Answer(
        f"{d['articles']:,} articles from {d['articles_from']} to {d['articles_to']} "
        f"across {len(d['sources'])} sources; {d['attributions']} attributions; "
        f"{len(d['tickers'])} tickers.")


def _llm_fallback(q: str) -> Answer:
    """Last resort: let the model phrase an answer over the stored summary.

    It is given ONLY material already computed, and is told to say it cannot
    answer rather than reason about the market.
    """
    from swing.agent.llm import get_llm, invoke_with_retry
    from swing.commands import coverage_data

    d = coverage_data()
    prompt = (
        "You answer questions about a stock-swing attribution database. You may "
        "only use the summary below. If it does not contain the answer, say so "
        "plainly. Never speculate about future prices, and never give investment "
        "advice.\n\n"
        f"Database: {d['articles']:,} articles {d['articles_from']}..{d['articles_to']}, "
        f"{d['attributions']} stored attributions, tickers: {', '.join(d['tickers'])}.\n\n"
        f"Question: {q}"
    )
    try:
        out = invoke_with_retry(get_llm(), prompt)
        return Answer(getattr(out, "content", str(out)), used_llm=True)
    except Exception as exc:  # noqa: BLE001 - quota exhaustion is expected on free tier
        return Answer(f"Could not reach the model ({type(exc).__name__}). "
                      "Try a specific query such as 'why did NVDA move'.")
