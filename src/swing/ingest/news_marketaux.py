"""Marketaux — ON-DEMAND ONLY, deliberately not polled.

Verified 2026-09-01 with a live key. The free tier returns 3 articles per
request at ~100 requests/day, and of 15 sampled articles exactly one survived
tiering (a CNBC piece already collected via RSS); the rest were gurufocus,
seekingalpha, thestockmarketwatch, investingcaffeine and insidermonkey.

Targeted domain queries returned ZERO tier-2 wire copy: reuters.com found=0,
apnews.com 0, bloomberg.com 0, wsj.com 0. Its documented purpose was as a dedup
stress test for syndicated wire copy, and there is no wire syndication in the
free stack to collapse.

Polling it would spend a daily quota to net ~20 duplicate CNBC articles. It is
useful for one thing: pulling extra breadth around ONE specific swing window
when investigating. That is what this module is for.
"""
from __future__ import annotations

from datetime import datetime

import httpx

from swing.common import logging as log
from swing.common.settings import get_settings
from swing.common.timeutil import assert_utc, parse_iso
from swing.store.raw import RawArticle, bump_health, insert_many

logger = log.get("ingest.marketaux")
BASE = "https://api.marketaux.com/v1/news/all"
SOURCE = "marketaux"


def fetch_window(tickers: list[str], start: datetime, end: datetime,
                 pages: int = 3) -> int:
    """Pull articles for specific tickers in a specific window.

    Bounded by `pages` because the quota is small: each page is one request and
    yields 3 articles. Default 3 pages = 3 requests = ~9 articles.
    """
    settings = get_settings()
    settings.require("marketaux_api_key")
    start, end = assert_utc(start), assert_utc(end)

    total = 0
    for page in range(1, pages + 1):
        try:
            r = httpx.get(BASE, params={
                "symbols": ",".join(t.upper() for t in tickers),
                "filter_entities": "true", "language": "en", "limit": 3, "page": page,
                "published_after": start.strftime("%Y-%m-%dT%H:%M"),
                "published_before": end.strftime("%Y-%m-%dT%H:%M"),
                "api_token": settings.marketaux_api_key,
            }, timeout=30)
        except Exception:
            logger.exception("marketaux page %d failed", page)
            break
        if r.status_code != 200:
            logger.warning("marketaux -> HTTP %s %s", r.status_code, r.text[:120])
            break

        data = r.json().get("data", [])
        if not data:
            break
        batch = []
        for a in data:
            if not (a.get("url") and a.get("title") and a.get("published_at")):
                continue
            batch.append(RawArticle(
                url=a["url"],
                # The PUBLISHER, so normalize.py tiers per publisher and drops
                # the tier-4 majority.
                source=(a.get("source") or SOURCE).strip().lower(),
                headline=a["title"],
                summary=a.get("description") or a.get("snippet") or None,
                published_at=parse_iso(a["published_at"]),
                raw={"publisher": a.get("source"), "via": SOURCE,
                     "entities": [e.get("symbol") for e in a.get("entities", [])],
                     "uuid": a.get("uuid")},
            ))
        n = insert_many(batch)
        total += n
    bump_health(SOURCE, total)
    logger.info("marketaux: %d new for %s", total, ",".join(tickers))
    return total


def fetch_for_swing(ticker: str, onset: datetime, hours_before: int = 24,
                    pages: int = 3) -> int:
    """Breadth for one swing's pre-move window."""
    from datetime import timedelta

    onset = assert_utc(onset)
    return fetch_window([ticker], onset - timedelta(hours=hours_before), onset, pages)
