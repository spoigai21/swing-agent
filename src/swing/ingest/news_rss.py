"""RSS/Atom poller for IR feeds (Tier 1) and market press (Tier 3).

RSS serves only the last 20-50 items, which is exactly why the collector has to
start before anything else gets built: an hour not polling is an hour of Tier 3
coverage that is gone permanently. agent-plan.md Phase -1.
"""
from __future__ import annotations

from typing import Any

import feedparser

from swing.common import logging as log
from swing.common.http import feed_get
from swing.common.timeutil import now_utc, parse_rss_datetime
from swing.ingest.config import FeedSpec
from swing.store.raw import RawArticle, bump_health, insert_many

logger = log.get("ingest.rss")


def _entry_text(entry: Any, *names: str) -> str | None:
    for n in names:
        v = entry.get(n)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and v:
            first = v[0]
            if isinstance(first, dict) and first.get("value"):
                return str(first["value"]).strip()
    return None


def _to_article(spec: FeedSpec, entry: Any) -> RawArticle | None:
    url = _entry_text(entry, "link", "id")
    headline = _entry_text(entry, "title")
    if not url or not headline:
        return None

    published = parse_rss_datetime(
        entry.get("published") or entry.get("updated"),
        entry.get("published_parsed") or entry.get("updated_parsed"),
    )
    if published is None:
        # No usable timestamp means no timing evidence, and timing is the whole
        # point. Drop rather than guess — a wrong published_at silently
        # reclassifies post-move commentary as a candidate catalyst.
        logger.debug("%s: dropping entry with unparseable date: %s", spec.id, headline[:60])
        return None
    if published > now_utc().replace(microsecond=0):
        # Feeds occasionally publish future-dated items; clamp rather than trust.
        published = now_utc()

    summary = _entry_text(entry, "summary", "description", "subtitle")
    return RawArticle(
        url=url,
        source=spec.source,
        headline=headline,
        summary=summary,
        published_at=published,
        raw={
            "feed_id": spec.id,
            "source_tier": spec.tier,
            "feed_tickers": spec.tickers,
            "entry": {k: v for k, v in entry.items() if isinstance(v, (str, int, float, list, dict))},
        },
    )


def poll_feed(spec: FeedSpec) -> int:
    """Fetch one feed and store new entries. Returns count actually new."""
    try:
        resp = feed_get(spec.url)
    except Exception as exc:
        logger.warning("%s: fetch failed: %s", spec.id, exc)
        bump_health(spec.source, 0)
        return 0

    if resp.status_code != 200:
        logger.warning("%s: HTTP %s", spec.id, resp.status_code)
        bump_health(spec.source, 0)
        return 0

    parsed = feedparser.parse(resp.content)
    if parsed.bozo and not parsed.entries:
        logger.warning("%s: unparseable feed (%s)", spec.id, parsed.get("bozo_exception"))
        bump_health(spec.source, 0)
        return 0

    articles = [a for a in (_to_article(spec, e) for e in parsed.entries) if a]
    n = insert_many(articles)
    bump_health(spec.source, n)
    if n:
        logger.info("%s: %d new / %d seen", spec.id, n, len(parsed.entries))
    else:
        logger.debug("%s: 0 new / %d seen", spec.id, len(parsed.entries))
    return n
