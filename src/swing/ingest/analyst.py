"""Analyst rating and price-target changes, from Yahoo Finance.

Broker upgrades, downgrades and target changes drive a real share of
idiosyncratic moves and often reach the tape before any article does
(data-sources.md C.4). Finnhub's upgrade/downgrade and price-target endpoints
are paid (403, verified 2026-09-14), and its free recommendation trend is a
monthly count that cannot place an action on a day.

Yahoo's `upgrades_downgrades` is free, reaches back years and is timestamped to
the second. GradeDate is naive UTC: verified 2026-09-14 against 208 of the same
actions as Benzinga reported them, which land 1-3 minutes later every time.
Each action becomes one synthetic Tier 2 article tagged to its stock.
"""
from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta

from swing.common import logging as log
from swing.ingest.config import stocks
from swing.store.raw import RawArticle, bump_health, insert_many

logger = log.get("ingest.analyst")
SOURCE = "analyst-ratings"

_VERBS = {"up": "upgrades", "down": "downgrades", "init": "initiates",
          "main": "maintains", "reit": "reiterates"}


def _num(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or v <= 0 else v


def _money(value) -> str | None:
    v = _num(value)
    if v is None:
        return None
    return f"${v:,.0f}" if v == int(v) else f"${v:,.2f}"


def headline(ticker: str, company: str, firm: str, action: str, to_grade: str,
             from_grade: str, target_action: str, target, prior_target) -> str:
    """One broker action as a sentence, e.g.
    'UBS upgrades Tesla (TSLA) to Neutral from Sell; price target $352'."""
    who = f"{company} ({ticker})"
    to_g, from_g = (to_grade or "").strip(), (from_grade or "").strip()
    if action in ("up", "down") and from_g:
        rating = f"{firm} {_VERBS[action]} {who} to {to_g} from {from_g}"
    elif action == "init":
        rating = f"{firm} initiates {who} at {to_g}" if to_g else f"{firm} initiates coverage of {who}"
    else:
        verb = _VERBS.get(action, "rates")
        rating = f"{firm} {verb} {to_g} on {who}" if to_g else f"{firm} {verb} {who}"

    new, old = _money(target), _money(prior_target)
    change = (target_action or "").strip().lower()
    if new and old and change in ("raises", "lowers"):
        return f"{rating}; {change} price target to {new} from {old}"
    return f"{rating}; price target {new}" if new else rating


def fetch(ticker: str, since: datetime | None = None) -> int:
    """Store every rating action for `ticker` at or after `since`. Returns new rows."""
    import pandas as pd
    import yfinance as yf

    df = yf.Ticker(ticker).upgrades_downgrades
    if df is None or df.empty:
        return 0
    # "Tesla", not "Tesla Inc": the first curated alias reads like a headline.
    company = ((stocks().get(ticker) or {}).get("aliases") or [ticker])[0]
    batch = []
    for ts, row in df.iterrows():
        at = pd.Timestamp(ts).to_pydatetime()
        at = at.replace(tzinfo=UTC) if at.tzinfo is None else at.astimezone(UTC)
        firm = str(row.get("Firm") or "").strip()
        if not firm or (since and at < since):
            continue
        action = str(row.get("Action") or "").strip()
        slug = re.sub(r"[^a-z0-9]+", "-", firm.lower()).strip("-")
        batch.append(RawArticle(
            url=f"yahoo-analyst://{ticker}/{slug}/{at:%Y%m%dT%H%M%SZ}",
            source=SOURCE,
            headline=headline(ticker, company, firm, action, str(row.get("ToGrade") or ""),
                              str(row.get("FromGrade") or ""),
                              str(row.get("priceTargetAction") or ""),
                              row.get("currentPriceTarget"), row.get("priorPriceTarget")),
            published_at=at,
            raw={"via": "yahoo-analyst", "ticker": ticker, "firm": firm, "action": action,
                 "to_grade": row.get("ToGrade"), "from_grade": row.get("FromGrade"),
                 "price_target_action": row.get("priceTargetAction"),
                 "price_target": _num(row.get("currentPriceTarget")),
                 "prior_price_target": _num(row.get("priorPriceTarget")),
                 "source_tier": 2},
        ))
    n = insert_many(batch)
    bump_health(SOURCE, n)
    if n:
        logger.info("analyst %s: %d new actions", ticker, n)
    return n


def poll(since_days: int = 30) -> int:
    """Recent rating actions for every watchlist stock."""
    return backfill(datetime.now(UTC) - timedelta(days=since_days))


def backfill(since: datetime) -> int:
    total = 0
    for ticker in stocks():
        try:
            total += fetch(ticker, since)
        except Exception:
            logger.exception("analyst ratings failed for %s", ticker)
    return total
