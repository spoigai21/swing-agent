"""Finnhub company news — the Tier 2 workhorse, one call per ticker.

Finnhub returns MANY publishers through one endpoint, mixing Tier 2 wire copy
with Tier 4 aggregator content (SeekingAlpha, Zacks, Benzinga...). The publisher
is carried per-article in `source`, so normalize.py tiers on that, not on the
feed. Without this, aggregator posts enter the evidence pool as if they were
wire reports. data-sources.md C.2.

Finnhub's own `sentiment` field is deliberately ignored: a weak sentiment score
in the pipeline is worse than none.
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import httpx

from swing.common import logging as log
from swing.ingest.config import stocks
from swing.store.raw import RawArticle, bump_health, insert_many

logger = log.get("ingest.finnhub")
BASE = "https://finnhub.io/api/v1"
SOURCE = "finnhub"


def _get(path: str, params: dict) -> httpx.Response:
    """Paced, 429-aware Finnhub GET.

    The free tier allows 60 calls/min. The previous version retried only on
    transport errors, so a 429 came back as a normal response, the chunk was
    skipped, and it looked like "no news existed then" rather than throttling.
    """
    from swing.common.http import finnhub_get

    return finnhub_get(path, params)


def poll(days: int = 7) -> int:
    """Company news for every watchlist ticker over the last `days`."""
    end = datetime.now(UTC).date()
    start = end - timedelta(days=days)
    total = 0

    for ticker in stocks():
        try:
            r = _get("/company-news", {"symbol": ticker,
                                       "from": start.isoformat(), "to": end.isoformat()})
        except Exception:
            logger.exception("finnhub news failed for %s", ticker)
            continue
        if r.status_code != 200:
            logger.warning("finnhub news %s -> HTTP %s", ticker, r.status_code)
            continue

        batch = []
        for item in r.json():
            url = item.get("url")
            headline = item.get("headline")
            ts = item.get("datetime")
            if not (url and headline and ts):
                continue
            batch.append(RawArticle(
                url=url,
                # The PUBLISHER, not 'finnhub'. normalize.py tiers on this.
                source=(item.get("source") or SOURCE).strip().lower(),
                headline=headline,
                summary=item.get("summary") or None,
                published_at=datetime.fromtimestamp(int(ts), UTC),
                raw={"publisher": item.get("source"), "finnhub_id": item.get("id"),
                     "category": item.get("category"), "related": item.get("related"),
                     "ticker": ticker, "via": SOURCE},
            ))
        n = insert_many(batch)
        bump_health(SOURCE, n)
        total += n
        if n:
            logger.info("finnhub %s: %d new / %d returned", ticker, n, len(r.json()))
    return total


def poll_recommendations() -> int:
    """Analyst recommendation trend as synthetic Tier 2 articles.

    Broker actions often hit the tape before any article does and drive a
    meaningful share of idiosyncratic moves for the semis. Without these you get
    a cluster of `unexplained` verdicts that were all broker actions.
    data-sources.md C.4.

    NOTE: /stock/price-target is 403 on the free tier, so only the
    recommendation trend is available. Coverage here is partial.
    """
    total = 0
    for ticker in stocks():
        try:
            r = _get("/stock/recommendation", {"symbol": ticker})
        except Exception:
            logger.exception("recommendation failed for %s", ticker)
            continue
        if r.status_code != 200:
            continue
        batch = []
        for row in r.json():
            period = row.get("period")
            if not period:
                continue
            headline = (f"{ticker} analyst recommendations {period}: "
                        f"{row.get('strongBuy',0)} strong buy, {row.get('buy',0)} buy, "
                        f"{row.get('hold',0)} hold, {row.get('sell',0)} sell, "
                        f"{row.get('strongSell',0)} strong sell")
            batch.append(RawArticle(
                url=f"finnhub://recommendation/{ticker}/{period}",
                source="finnhub-recommendation",
                headline=headline,
                published_at=datetime.fromisoformat(period).replace(tzinfo=UTC),
                raw={"ticker": ticker, "source_tier": 2, "kind": "recommendation_trend", **row},
            ))
        n = insert_many(batch)
        bump_health("finnhub-recommendation", n)
        total += n
    return total


def backfill(start: date, end: date, chunk_days: int = 7,
             tickers: list[str] | None = None) -> int:
    """Walk company-news backwards in chunks. THE way to unblock Gate 2.

    RSS cannot be backfilled — it serves only the last 20-50 items — but
    Finnhub's company-news endpoint DOES accept historical date ranges on the
    free tier, verified 2026-09-03 back to roughly 2025-09 (about 12 months;
    2025-08 and earlier return nothing).

    That matters because Gate 2 needs blind annotations, and those can only
    cover dates where we hold articles. Without this you wait months for
    coverage to accumulate; with it, a year of it arrives in minutes.

    Expect a low keep rate: most Finnhub content is tier 4 and is dropped at
    normalize. The archive keeps everything regardless.
    """
    syms = tickers or list(stocks())
    total = 0
    cur = end
    while cur > start:
        frm = max(start, cur - timedelta(days=chunk_days))
        for ticker in syms:
            try:
                r = _get("/company-news", {"symbol": ticker,
                                           "from": frm.isoformat(), "to": cur.isoformat()})
            except Exception:
                logger.exception("backfill failed for %s %s..%s", ticker, frm, cur)
                continue
            if r.status_code != 200:
                logger.warning("backfill %s %s -> HTTP %s", ticker, frm, r.status_code)
                continue
            batch = []
            for item in r.json():
                url, headline, ts = item.get("url"), item.get("headline"), item.get("datetime")
                if not (url and headline and ts):
                    continue
                batch.append(RawArticle(
                    url=url,
                    source=(item.get("source") or SOURCE).strip().lower(),
                    headline=headline,
                    summary=item.get("summary") or None,
                    published_at=datetime.fromtimestamp(int(ts), UTC),
                    raw={"publisher": item.get("source"), "finnhub_id": item.get("id"),
                         "category": item.get("category"), "ticker": ticker,
                         "via": SOURCE, "backfill": True},
                ))
            total += insert_many(batch)
        logger.info("backfill %s..%s -> %d stored so far", frm, cur, total)
        cur = frm
    bump_health(SOURCE, 0)
    return total
