"""Answer "why did this stock move?" from a single question.

This is the product as the user sees it: type `swing`, ask why a stock went up
or down, get the reason with its sources. Each question refreshes exactly what
it needs, so an answer never depends on a batch job having run:

  latest prices -> split the move into market / sector / the stock's own part
    -> own part within its normal range: say so, no model call
    -> unusual: find when the move started, gather the news published BEFORE
       it, ask Gemini, and let the code-level guards (citations must exist,
       no pre-move evidence means no story) decide what reaches the user

Watchlist stocks only, for now.
"""
from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from swing.common import logging as log
from swing.interface.guardrail import check as guard_check

# Library chatter (download progress, HF auth notices) must not bury the answer.
for _key, _val in (("HF_HUB_DISABLE_PROGRESS_BARS", "1"), ("HF_HUB_VERBOSITY", "error"),
                   ("TRANSFORMERS_VERBOSITY", "error"), ("TOKENIZERS_PARALLELISM", "false")):
    os.environ.setdefault(_key, _val)

logger = log.get("interface.explain")

ET = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_FINAL = time(16, 15)        # the daily bar settles shortly after the close

# ⚠️ Display names live in config (sources.yaml `source_labels`, watchlist.yaml
# `label` per sector), not here. Hand-written copies in code meant adding a feed
# or a sector silently degraded the answer — the same drift that had
# `primary_sources` written out twice.

# Questions about the stored history rather than one stock's move; query.py
# answers those with SQL.
_HISTORY_INTENT = re.compile(r"\b(unexplained|idio\w*|biggest|largest|coverage|how many)\b",
                             re.IGNORECASE)
# Capitalised words that are not ticker symbols.
_NOT_TICKERS = {"AI", "US", "USA", "ET", "CEO", "CFO", "ETF", "IPO", "SEC", "FDA", "EPS",
                "GDP", "CPI", "FED", "OK", "EV", "WHY", "WHAT", "HOW", "DID", "IS"}
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]


def _say(msg: str) -> None:
    print(f"  · {msg}", flush=True)


# --------------------------------------------------------------------------
# Understanding the question
# --------------------------------------------------------------------------

@dataclass(slots=True)
class Question:
    ticker: str | None = None
    day: date | None = None             # None = the latest completed session
    unknown_symbol: str | None = None


def _prior_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def last_completed_session(now: datetime) -> date:
    """The latest session whose daily bar is final. Weekends are skipped here;
    holidays resolve later to the latest bar on or before this date."""
    et = now.astimezone(ET)
    if et.weekday() < 5 and et.time() >= SESSION_FINAL:
        return et.date()
    return _prior_weekday(et.date() - timedelta(days=1))


def market_open(now: datetime) -> bool:
    et = now.astimezone(ET)
    return et.weekday() < 5 and SESSION_OPEN <= et.time() < SESSION_FINAL


def _parse_day(question: str, today: date) -> date | None:
    low = question.lower()
    if m := re.search(r"\b(\d{4}-\d{2}-\d{2})\b", question):
        return date.fromisoformat(m.group(1))
    if "yesterday" in low:
        return _prior_weekday(today - timedelta(days=1))
    if re.search(r"\btoday\b", low):
        return today
    if m := re.search(rf"\b({'|'.join(_MONTHS)})[a-z]*\.?\s+(\d{{1,2}})\b", low):
        try:
            d = date(today.year, _MONTHS.index(m.group(1)) + 1, int(m.group(2)))
        except ValueError:
            return None
        return d if d <= today else d.replace(year=today.year - 1)
    for i, name in enumerate(_WEEKDAYS):
        if re.search(rf"\b{name}\b", low):
            return today - timedelta(days=(today.weekday() - i) % 7 or 7)
    return None


def parse(question: str, today: date) -> Question:
    """Find the stock and the day a question is about."""
    from swing.ingest.config import stocks

    known = stocks()
    day = _parse_day(question, today)
    for tok in re.findall(r"\b[A-Za-z]{2,5}\b", question):
        # MU is also an ordinary word, so a symbol must be in capitals unless it
        # is long enough to be unambiguous (nvda, tsla).
        if tok.upper() in known and (tok.isupper() or len(tok) >= 4):
            return Question(tok.upper(), day)
    low = question.lower()
    for sym, meta in known.items():
        for alias in meta.get("aliases") or []:
            if re.search(rf"\b{re.escape(alias.lower())}\b", low):
                return Question(sym, day)
    for tok in re.findall(r"\b[A-Z]{2,5}\b", question):
        if tok not in _NOT_TICKERS:
            return Question(unknown_symbol=tok)
    return Question()


def respond(question: str, *, now: datetime | None = None,
            say: Callable[[str], None] = _say) -> str:
    """One question in, one answer out. The forecast guardrail runs first."""
    g = guard_check(question)
    if not g.allowed:
        return g.reason or "refused"
    now = now or datetime.now(UTC)
    if not _HISTORY_INTENT.search(question):
        q = parse(question, now.astimezone(ET).date())
        if q.ticker:
            return explain(q.ticker, q.day, now=now, say=say)
        if q.unknown_symbol:
            from swing.ingest.config import stocks

            return (f"I only cover these stocks for now: {', '.join(sorted(stocks()))}.\n"
                    f"{q.unknown_symbol} isn't one of them yet.")
    from swing.interface.query import answer

    return answer(question).text


# --------------------------------------------------------------------------
# Answering it
# --------------------------------------------------------------------------

def explain(ticker: str, day: date | None = None, *, now: datetime | None = None,
            say: Callable[[str], None] = _say) -> str:
    """Why `ticker` moved on `day` (default: the latest completed session)."""
    from swing.analysis.factors import entities, rebuild
    from swing.ingest.config import thresholds
    from swing.ingest.prices import refresh_daily

    now = now or datetime.now(UTC)
    if day is not None and day.weekday() >= 5:
        return f"The market was closed on {_day(day)} (weekend), so there's no move to explain."
    entity = next(e for e in entities() if e.ticker == ticker)
    today_et = now.astimezone(ET).date()

    say("checking latest prices")
    refresh_daily([s for s in (ticker, entity.sector_etf, entity.market_etf) if s])
    rebuild(ticker)

    live = _so_far_today(ticker, now) if market_open(now) else None
    if day is not None and day >= today_et and day > last_completed_session(now):
        return (f"{_day(day)}'s session isn't finished yet, so there's no final move to "
                "explain." + (live or ""))

    dec = _decomposition(ticker, day or last_completed_session(now), exact=day is not None)
    if dec is None:
        return (f"No trading data for {ticker} on {_day(day)}. The market may have been closed."
                if day else f"No price data for {ticker} yet.")
    if dec.status != "ok":
        return (f"{ticker} doesn't have enough trading history yet to separate its own move "
                "from the market's, so I can't explain it reliably.")

    lines = [header(dec, entity.sector_etf)]
    if abs(dec.residual_z) < float(thresholds()["swings"]["z_threshold"]):
        lines.append(normal_day(dec, entity.sector_etf))
        if context := _drift_context(ticker, dec.d):
            lines.append(context)
    else:
        lines.append(_unusual_day(entity, dec, say))
    if live and day is None:
        lines.append(live)
    return "\n".join(lines)


def _drift_context(ticker: str, d: date) -> str | None:
    """A day can be ordinary on its own and still sit inside a flagged drift.

    Detection uses a CUMULATIVE z over 3-5 days (`drift_z_threshold`, 2.5);
    this function's caller uses the SINGLE day (`z_threshold`, 2.0). Both are
    right, and together they let the product say "nothing unusual" about a date
    that is in the swings table and may already have fired an alert — which
    reads as the system contradicting itself. MRVL 2026-09-02: ordinary alone
    at 1.2x typical, but the end of a drift at z=-2.55.

    Costs no model call: this is context on the normal-day answer, not an
    attribution.
    """
    from swing.store.session import connect

    with connect() as conn:
        row = conn.execute(
            "SELECT d, drift_window, total_return, residual_z FROM swings "
            "WHERE ticker = %s AND kind = 'drift' AND drift_window IS NOT NULL "
            "  AND %s BETWEEN d - (drift_window - 1) AND d "
            "ORDER BY abs(residual_z) DESC LIMIT 1",
            (ticker, d)).fetchone()
    if not row:
        return None
    return (f"\nWorth knowing: this day sits inside a flagged {row['drift_window']}-day "
            f"run ending {_day(row['d'])} — {row['total_return']:+.1%} cumulative, "
            f"{abs(row['residual_z']):.1f} sigma. Ordinary day by day; the run is not.")


def _unusual_day(entity, dec, say: Callable[[str], None]) -> str:
    from swing.agent.graph import attribute_swing
    from swing.analysis.retrieval import build_for_swing
    from swing.analysis.swings import detect_day

    say("finding when the move started")
    swing_id = detect_day(entity, dec.d)
    if swing_id is None:
        return normal_day(dec, entity.sector_etf)
    say("gathering news from before the move")
    _refresh_news(entity.ticker, dec.d)
    build_for_swing(swing_id)

    stored = _reusable_attribution(swing_id)
    if stored:
        verdict, payload, note = stored["verdict"], stored["payload"], stored["unexplained_note"]
    else:
        say("asking Gemini")
        out = attribute_swing(swing_id)
        attr = out.get("attribution")
        if attr is None or out.get("verdict_reason") == "llm_error":
            return model_unavailable(_pre_move_evidence(swing_id))
        verdict, payload, note = attr.verdict, attr.model_dump(), attr.unexplained_note
    return render_verdict(verdict, payload, note, _cluster_details(payload))


def _decomposition(ticker: str, target: date, exact: bool):
    from swing.analysis.decompose import load
    from swing.store.session import connect

    with connect() as conn:
        r = conn.execute(
            "SELECT d, status FROM daily_factors WHERE ticker=%s AND d <= %s "
            "ORDER BY d DESC LIMIT 1", (ticker, target)).fetchone()
    if r and r["status"] != "ok":
        return load(ticker, r["d"])             # not enough history: say so, whatever the day
    if exact:
        return load(ticker, target)
    return load(ticker, r["d"]) if r else None


def _refresh_news(ticker: str, d: date) -> None:
    """Pull what the collector may not hold yet: SEC filings (the stock's and its
    related companies'), analyst actions, and news for the stock and each
    related company."""
    from swing.ingest import analyst, edgar, gdelt, news_finnhub
    from swing.ingest.config import related
    from swing.ingest.normalize import normalize_all

    start, end = d - timedelta(days=3), d + timedelta(days=1)
    since = datetime.combine(d - timedelta(days=7), time(), tzinfo=UTC)
    # Scoped to this question's companies: unfiltered, EDGAR is one request per
    # CIK and dominated question latency. The collector still polls them all.
    jobs = [("SEC filings", lambda: edgar.poll(tickers=[ticker, *related(ticker)])),
            ("analyst ratings", lambda: analyst.fetch(ticker, since)),
            ("company news", lambda: news_finnhub.fetch(ticker, start, end))]
    jobs += [(f"{t} news", lambda t=t: news_finnhub.fetch(t, start, end))
             for t in related(ticker)]
    # Wire copy (Reuters, Bloomberg, WSJ, FT) that no free API carries. The
    # collector's 4h pass covers recent days; this fills older windows and the
    # last few hours. gdelt.refresh is a no-op once the window is stored, so
    # asking the same question twice does not pay for the same bytes twice.
    jobs.append(("wire copy", lambda: gdelt.refresh(
        [ticker, *related(ticker)],
        datetime.combine(start, time(), tzinfo=UTC),
        datetime.combine(end, time(23, 59, 59), tzinfo=UTC))))
    for name, fn in jobs:
        try:
            fn()
        except Exception:
            logger.warning("refreshing %s for %s failed", name, ticker, exc_info=True)
    normalize_all()
    try:
        from swing.ingest.edgar_text import enrich_pending

        enrich_pending(limit=20)            # new filings: read their press releases
    except Exception:
        logger.warning("filing text refresh failed", exc_info=True)


def _reusable_attribution(swing_id: int) -> dict | None:
    """A stored answer built from exactly the evidence we would pass now, by the
    same model, prompt and config.

    ⚠️ This is a QUOTA CACHE, not a determinism shortcut. Gemini 3.x ignores
    temperature, so asking again would spend one of 20 daily requests to get a
    possibly differently-worded answer — reuse keeps the answer stable for the
    same evidence, which is the behaviour you want anyway."""
    from swing.common.versioning import config_hash, model_id, prompt_version
    from swing.ingest.config import thresholds
    from swing.store.session import connect

    top_k = int(thresholds()["retrieval"]["max_clusters_to_llm"])
    with connect() as conn:
        shown = {r["id"] for r in conn.execute(
            "SELECT id FROM (SELECT id, row_number() OVER (PARTITION BY timing ORDER BY rank) n "
            "FROM clusters WHERE swing_id=%s) x WHERE n <= %s", (swing_id, top_k)).fetchall()}
        rows = conn.execute(
            "SELECT verdict, payload, unexplained_note, shown_cluster_ids FROM attributions "
            "WHERE swing_id=%s AND run_kind='production' AND model_id=%s "
            "AND prompt_version=%s AND config_hash=%s ORDER BY created_at DESC",
            (swing_id, model_id(), prompt_version(), config_hash())).fetchall()
    return next((r for r in rows if set(r["shown_cluster_ids"] or []) == shown), None)


def _cluster_details(payload: dict) -> dict[int, dict]:
    from swing.store.session import connect

    ids = [e["cluster_id"] for c in payload.get("candidates") or []
           for e in c.get("evidence") or []]
    if not ids:
        return {}
    with connect() as conn:
        rows = conn.execute(
            "SELECT c.id, c.earliest_published, a.source, a.headline, a.url "
            "FROM clusters c JOIN articles a ON a.id = c.canonical_article "
            "WHERE c.id = ANY(%s)", (ids,)).fetchall()
    return {r["id"]: dict(r) for r in rows}


def _pre_move_evidence(swing_id: int) -> list[dict]:
    from swing.store.session import connect

    with connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT c.earliest_published, a.source, a.headline FROM clusters c "
            "JOIN articles a ON a.id = c.canonical_article "
            "WHERE c.swing_id=%s AND c.timing='pre_move' ORDER BY c.rank LIMIT 3",
            (swing_id,)).fetchall()]


def _so_far_today(ticker: str, now: datetime) -> str | None:
    from swing.store.session import connect

    with connect() as conn:
        rows = conn.execute("SELECT ts::date AS d, close FROM bars WHERE ticker=%s "
                            "ORDER BY ts DESC LIMIT 2", (ticker,)).fetchall()
    if len(rows) < 2 or rows[0]["d"] != now.astimezone(ET).date():
        return None
    change = float(rows[0]["close"]) / float(rows[1]["close"]) - 1
    return (f"\n(Today's session is still open: {ticker} is {_pct(change)} so far. "
            "Ask again after 4:15pm ET for today's reason.)")


# --------------------------------------------------------------------------
# Wording. Pure functions, so every phrasing is tested.
# --------------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _day(d: date) -> str:
    return f"{d:%a %b} {d.day}"


def _when(ts: datetime) -> str:
    et = ts.astimezone(ET)
    return f"{_day(et.date())}, {et.hour % 12 or 12}:{et:%M}{'am' if et.hour < 12 else 'pm'} ET"


def _source_label(source: str) -> str:
    from swing.ingest.config import source_labels

    label = source_labels().get(source)
    if label:
        return label
    return "company press release" if source.endswith("-ir") else source.title()


def _sector(sector_etf: str | None) -> str | None:
    from swing.ingest.config import sector_label

    return sector_label(sector_etf) if sector_etf else None


def balanced_pcts(values: list[float], total: float) -> list[float]:
    """Round to 0.1% so the parts still sum to the displayed total.

    ⚠️ Rounding each part on its own is what makes a correct answer look wrong:
    four terms can each be off by 0.05 and miss the rounded total by 0.2, and a
    tool whose pitch is "check my arithmetic" cannot print a split that does not
    add up.

    Largest-remainder allocation, in integer tenths of a percent: every part is
    rounded DOWN, then the leftover tenths go one at a time to the parts with
    the biggest fractions. A naive "dump the whole leftover on one part" was the
    first attempt and moved a term by 0.14% — more than a display step, i.e. a
    number that is simply wrong rather than merely rounded.
    """
    import math

    target = round(total * 1000)                 # tenths of a percent
    floors = [math.floor(v * 1000) for v in values]
    rema = [v * 1000 - f for v, f in zip(values, floors)]
    leftover = target - sum(floors)
    order = sorted(range(len(values)), key=lambda i: rema[i], reverse=leftover > 0)
    for k in range(min(abs(leftover), len(values))):
        floors[order[k]] += 1 if leftover > 0 else -1
    return [f / 10 for f in floors]


def split_terms(dec, sector_etf: str | None) -> list[tuple[str, float]]:
    """Every term of the decomposition, including the one that used to be hidden.

    total = drift + market + sector + residual. The regression intercept — the
    stock's average daily drift over the fit window — was never displayed, so
    NFLX on 2026-09-21 printed "market +0.4% · comm stocks +3.2% · NFLX on its
    own -1.1%" for a +2.2% day: right to four decimal places, and visibly wrong
    to anyone adding it up.

    It stays a separate term rather than being folded into the stock's own part,
    because the residual alone is what `residual_z` and swing detection are
    computed from. Folding it in would make the printed number disagree with the
    number the rest of the sentence is about.
    """
    drift = (dec.total_return - dec.market_component
             - dec.sector_component - dec.residual)
    terms = [("market", dec.market_component)]
    if sector_etf:
        terms.append((_sector(sector_etf), dec.sector_component))
    # Below 0.05% it rounds to +0.0% and the other terms already add up.
    if abs(round(drift * 100, 1)) >= 0.1:
        terms.append(("its usual drift", drift))
    terms.append((f"{dec.ticker} on its own", dec.residual))
    return terms


def header(dec, sector_etf: str | None) -> str:
    if abs(dec.total_return) < 0.0005:
        move = f"{dec.ticker} was flat on {_day(dec.d)}."
    else:
        verb = "rose" if dec.total_return > 0 else "fell"
        move = f"{dec.ticker} {verb} {abs(dec.total_return) * 100:.1f}% on {_day(dec.d)}."
    terms = split_terms(dec, sector_etf)
    shown = balanced_pcts([v for _, v in terms], dec.total_return)
    parts = [f"{label} {v:+.1f}%" for (label, _), v in zip(terms, shown)]
    return f"{move}\n  Split: {' · '.join(parts)}"


def notable_company_news(ticker: str, day, limit: int = 3) -> list[dict]:
    """Company-specific items on a day no swing was flagged.

    ⚠️ Added after `swing why NFLX --date 2026-09-22` said "there's no
    company-specific news to find" on a day HSBC had downgraded the stock. The
    article was in the corpus; the move simply was not unusual. Saying there is
    no news, when what is true is that there is no unusual MOVE, is a claim the
    tool cannot support and a user can immediately disprove.
    """
    from swing.ingest.config import primary_sources
    from swing.store.session import connect

    with connect() as conn:
        return [dict(r) for r in conn.execute(
            """
            SELECT published_at, source, headline, url
            FROM articles
            WHERE %s = ANY(tickers)
              AND published_at >= %s::date AND published_at < %s::date + 1
              AND (source = ANY(%s) OR source LIKE %s)
            ORDER BY source_tier, published_at
            LIMIT %s
            """,
            (ticker, day, day, primary_sources(), "%-ir", limit)).fetchall()]


def _quiet_day_news(dec) -> str:
    """Name what the company did, without claiming it moved the stock."""
    try:
        items = notable_company_news(dec.ticker, dec.d)
    except Exception:   # noqa: BLE001 — a nicety must never break the answer
        return ""
    if not items:
        return ""
    lines = ["\n  There was company news, but the move does not need it:"]
    lines += [f"  • {_when(i['published_at'])} · {_source_label(i['source'])}"
              f" · {i['headline'][:76]}" for i in items]
    return "\n".join(lines)


def normal_day(dec, sector_etf: str | None) -> str:
    size = f"about {abs(dec.residual_z):.1f}x a typical day for {dec.ticker}"
    factors = dec.market_component + dec.sector_component
    drivers = "the market" + (f" and {_sector(sector_etf)}" if sector_etf else "")
    # ⚠️ Every branch says there is no unusual MOVE — never that there is no
    # news. Those are different claims, and only the first one is measured.
    if abs(dec.residual) <= 0.5 * abs(factors):
        head = (f"\nWhy: mostly {drivers}. {dec.ticker}'s own part is {size}, which is "
                "normal, so there is no company-specific move to explain.")
    elif factors * dec.residual < 0:
        # The stock went against its factors: say that, not "mostly the market".
        verb = "lagged" if dec.residual < 0 else "beat"
        head = (f"\nWhy: {dec.ticker} {verb} {drivers} by {abs(dec.residual) * 100:.1f}%. "
                f"That gap is {size}, within its normal range, so there is no "
                "company-specific move to explain.")
    else:
        head = (f"\nWhy: nothing unusual. {dec.ticker}'s own move is {size}, within its "
                "normal range, so there is no company-specific move to explain.")
    return head + _quiet_day_news(dec)


def render_verdict(verdict: str, payload: dict, note: str | None,
                   details: dict[int, dict]) -> str:
    from swing.agent.abstention import NOTE_BASE

    candidates = payload.get("candidates") or []
    if verdict == "unexplained" or not candidates:
        rest = (note or "").removeprefix(NOTE_BASE).strip()
        return ("\nWhy: no news published before the move explains it."
                + (f"\n  {rest}" if rest else ""))

    top = candidates[0]
    lead = "Why" if verdict == "explained" else "Partly explained"
    lines = [f"\n{lead}: {top['catalyst']}"]
    seen: set[int] = set()
    for e in top.get("evidence") or []:
        d = details.get(e["cluster_id"])
        if d is None or e["cluster_id"] in seen:
            continue                            # only evidence we actually hold is shown
        seen.add(e["cluster_id"])
        lines.append(f"  • {_when(d['earliest_published'])} · {_source_label(d['source'])}"
                     f" · {d['headline'][:90]}")
        if (d.get("url") or "").startswith("http"):
            lines.append(f"    {d['url']}")
    lines += [f"  Also possible: {c['catalyst']}" for c in candidates[1:]]
    lines.append(f"  Confidence: {top['confidence']}"
                 + ("" if verdict == "explained"
                    else " (the evidence is weak for a move this large)"))
    if payload.get("source_disagreement"):
        lines.append(f"  Outlets disagree: {payload['source_disagreement']}")
    return "\n".join(lines)


_WHY_NO_ANSWER = {
    "invalid_key": ("\nGoogle rejected your Gemini API key, so there's no checked reason. "
                    "Replace it with:  swing key"),
    "daily_cap": ("\nThe Gemini free tier's 20 requests for today are used up, so there's "
                  "no checked reason yet. It resets at midnight Pacific."),
}


def model_unavailable(evidence: list[dict]) -> str:
    from swing.agent.llm import last_failure

    lines = [_WHY_NO_ANSWER.get(last_failure() or "",
             "\nGemini didn't answer (it may be overloaded), so there's no checked "
             "reason yet. Ask again in a few minutes.")]
    if evidence:
        lines.append("  News from before the move, most relevant first (not yet checked):")
        lines += [f"  • {_when(r['earliest_published'])} · {_source_label(r['source'])}"
                  f" · {r['headline'][:90]}" for r in evidence]
    return "\n".join(lines)
