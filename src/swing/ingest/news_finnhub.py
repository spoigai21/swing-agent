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

from datetime import UTC, datetime, timedelta

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from swing.common import logging as log
from swing.common.settings import get_settings
from swing.ingest.config import stocks
from swing.store.raw import RawArticle, bump_health, insert_many

logger = log.get("ingest.finnhub")
BASE = "https://finnhub.io/api/v1"
SOURCE = "finnhub"


@retry(retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError)),
       wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(4),
       reraise=True)
def _get(path: str, params: dict) -> httpx.Response:
    settings = get_settings()
    settings.require("finnhub_api_key")
    return httpx.get(BASE + path, params={**params, "token": settings.finnhub_api_key},
                     timeout=30)


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
