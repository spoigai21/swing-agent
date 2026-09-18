"""Step 1.7 — characterise an earnings move instead of suppressing it.

agent-plan.md: "An earlier draft said to exclude earnings dates. **That was
wrong.** Earnings are the single largest driver of idiosyncratic moves —
suppressing them means going dark on the four days a year per ticker that matter
most." On a swing coinciding with an 8-K Item 2.02 the agent's job changes from
"find the catalyst" — already known — to "characterise it".

Consensus estimates are paid, so this does NOT compute surprise versus
expectations and must never imply it. What is free is the company's own filed
history: SEC XBRL `companyfacts` gives revenue, gross profit and diluted EPS per
quarter, from which sequential and year-over-year deltas follow.

    https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json

⚠️ Quarterly facts only. A 10-K reports the full year, so a naive read makes Q4
look like a 4x revenue explosion. Everything here is filtered to ~one-quarter
durations (80-100 days), which means **Q4 is missing for issuers that only
report it annually** — `deltas()` returns None rather than inventing it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from swing.common import logging as log
from swing.common.settings import REPO_ROOT

logger = log.get("analysis.earnings")

CACHE = REPO_ROOT / "data" / "xbrl"
CACHE_TTL_DAYS = 7          # filings are quarterly; a week-old cache is current
QUARTER_MIN_DAYS = 80
QUARTER_MAX_DAYS = 100

# Issuers tag revenue under either concept; check the specific one first.
REVENUE_CONCEPTS = ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues")
EPS_CONCEPT = "EarningsPerShareDiluted"
GROSS_PROFIT_CONCEPT = "GrossProfit"


@dataclass(frozen=True, slots=True)
class Quarter:
    start: date
    end: date
    value: float

    @property
    def days(self) -> int:
        return (self.end - self.start).days


@dataclass(frozen=True, slots=True)
class Deltas:
    ticker: str
    period_end: date
    revenue: float | None
    revenue_qoq: float | None
    revenue_yoy: float | None
    eps: float | None
    eps_qoq: float | None
    gross_margin: float | None
    gross_margin_prior: float | None

    def sentence(self) -> str:
        """One line for the prompt. Only states what was actually filed."""
        bits: list[str] = []
        if self.revenue is not None:
            move = f"${self.revenue / 1e9:.2f}B"
            if self.revenue_yoy is not None:
                move += f", {self.revenue_yoy:+.0%} YoY"
            if self.revenue_qoq is not None:
                move += f", {self.revenue_qoq:+.0%} QoQ"
            bits.append(f"Revenue {move}")
        if self.gross_margin is not None:
            gm = f"Gross margin {self.gross_margin:.1%}"
            if self.gross_margin_prior is not None:
                gm += f" (prior quarter {self.gross_margin_prior:.1%})"
            bits.append(gm)
        if self.eps is not None:
            eps = f"Diluted EPS ${self.eps:.2f}"
            if self.eps_qoq is not None:
                eps += f", {self.eps_qoq:+.0%} QoQ"
            bits.append(eps)
        if not bits:
            return ""
        # ⚠️ The disclaimer is not decoration: without it a reader assumes these
        # deltas are a beat or a miss, which needs consensus we do not have.
        return (". ".join(bits) + ". Company-filed figures only; no consensus "
                "comparison, so this is not a beat or a miss.")


def _cache_path(cik: str) -> Path:
    return CACHE / f"CIK{int(cik):010d}.json"


def companyfacts(cik: str, *, max_age_days: int = CACHE_TTL_DAYS) -> dict:
    """SEC XBRL company facts, cached on disk (each payload is several MB)."""
    from swing.common.http import sec_get

    path = _cache_path(cik)
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_days * 86400:
        try:
            return json.loads(path.read_text())
        except ValueError:
            logger.warning("corrupt companyfacts cache for %s; refetching", cik)

    resp = sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json")
    resp.raise_for_status()
    data = resp.json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def quarters(facts: dict, concept: str) -> list[Quarter]:
    """Quarterly-duration values for a us-gaap concept, oldest first.

    Annual durations are dropped: a 10-K's full-year revenue alongside quarterly
    figures would read as a 4x jump.
    """
    node = (facts.get("facts", {}).get("us-gaap", {}) or {}).get(concept)
    if not node:
        return []
    seen: dict[tuple[date, date], Quarter] = {}
    for entries in node.get("units", {}).values():
        for e in entries:
            if not e.get("start") or not e.get("end") or e.get("val") is None:
                continue
            try:
                start = date.fromisoformat(e["start"])
                end = date.fromisoformat(e["end"])
            except ValueError:
                continue
            if not QUARTER_MIN_DAYS <= (end - start).days <= QUARTER_MAX_DAYS:
                continue
            # Later amendments restate the same period; keep the last one filed.
            seen[(start, end)] = Quarter(start=start, end=end, value=float(e["val"]))
    return sorted(seen.values(), key=lambda q: q.end)


# A filed period must be close to the filing that reported it. Without this a
# stale series is served with full confidence: NVDA's specific-concept rows stop
# in 2020, so "first concept with any rows" returned a $3.10B quarter ending
# 2020-01-26 for a 2026 filing whose press release said $96.2 billion.
#
# ⚠️ 100 days, not 200. A quarterly report covers a period ending ~30 days before
# it is filed, so a gap beyond one quarter means the quarter being announced is
# NOT in the quarterly-duration data — the Q4-reported-annually case. SNDK filed
# 2026-08-05 announcing "$8.97 billion" for its fiscal Q4, and a 200-day window
# happily returned the April quarter's $5.95B in its place. Returning nothing is
# right there: a wrong number presented as this quarter's is worse than silence.
MAX_PERIOD_LAG_DAYS = 100


def _revenue(facts: dict, as_of: date | None = None) -> list[Quarter]:
    """The revenue series that actually covers `as_of`.

    ⚠️ Issuers disagree about which tag to use, and each abandons the other
    mid-history, so "first non-empty concept" picks a dead series half the time:

        NVDA  Revenues current to 2026; the specific concept stops in 2020
        AAPL  the specific concept current to 2026; Revenues stops in 2018
        SNDK  only the specific concept has quarterly rows
        QCOM  only Revenues has quarterly rows

    Choose whichever has coverage nearest `as_of` instead.
    """
    series = [rows for c in REVENUE_CONCEPTS if (rows := quarters(facts, c))]
    if not series:
        return []
    if as_of is None:
        return max(series, key=lambda rows: rows[-1].end)
    usable = [rows for rows in series if any(q.end <= as_of for q in rows)]
    if not usable:
        return []
    return max(usable, key=lambda rows: max(q.end for q in rows if q.end <= as_of))


def _pct(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return (new - old) / abs(old)


def _at_or_before(rows: list[Quarter], as_of: date) -> Quarter | None:
    return next((q for q in reversed(rows) if q.end <= as_of), None)


def _year_earlier(rows: list[Quarter], q: Quarter) -> Quarter | None:
    """The same fiscal quarter a year before, matched by nearest end date."""
    target = date(q.end.year - 1, q.end.month, min(q.end.day, 28))
    candidates = [r for r in rows if abs((r.end - target).days) <= 45]
    return min(candidates, key=lambda r: abs((r.end - target).days), default=None)


def deltas(ticker: str, as_of: date) -> Deltas | None:
    """Sequential and year-over-year deltas for the quarter reported by `as_of`."""
    from swing.ingest.config import cik_map

    cik = cik_map().get(ticker.upper())
    if not cik:
        return None
    try:
        facts = companyfacts(cik)
    except Exception:
        logger.warning("companyfacts unavailable for %s", ticker, exc_info=True)
        return None

    rev = _revenue(facts, as_of)
    eps, gp = quarters(facts, EPS_CONCEPT), quarters(facts, GROSS_PROFIT_CONCEPT)
    latest = _at_or_before(rev, as_of)
    # Refuse a stale quarter rather than presenting it as the one just reported.
    if latest is None or (as_of - latest.end).days > MAX_PERIOD_LAG_DAYS:
        return None
    prior = next((q for q in reversed(rev) if q.end < latest.end), None)
    year_ago = _year_earlier(rev, latest)

    eps_now = _at_or_before(eps, as_of)
    eps_prior = next((q for q in reversed(eps) if eps_now and q.end < eps_now.end), None)
    gp_now = next((q for q in gp if q.end == latest.end), None)
    gp_prior = next((q for q in gp if prior and q.end == prior.end), None)

    return Deltas(
        ticker=ticker.upper(),
        period_end=latest.end,
        revenue=latest.value,
        revenue_qoq=_pct(latest.value, prior.value if prior else None),
        revenue_yoy=_pct(latest.value, year_ago.value if year_ago else None),
        eps=eps_now.value if eps_now else None,
        eps_qoq=_pct(eps_now.value if eps_now else None,
                     eps_prior.value if eps_prior else None),
        gross_margin=(gp_now.value / latest.value) if gp_now and latest.value else None,
        gross_margin_prior=((gp_prior.value / prior.value)
                            if gp_prior and prior and prior.value else None),
    )


def characterize(ticker: str, as_of: date | datetime) -> str:
    """The earnings_mode line for the prompt, or "" when nothing is filed yet."""
    day = as_of.date() if isinstance(as_of, datetime) else as_of
    d = deltas(ticker, day)
    return d.sentence() if d else ""
