"""Per-source daily counts and the dead-feed alert.

A silently dead feed for three weeks is three weeks of unrecoverable data, and
it does not announce itself. agent-plan.md Step -1.3.
"""
from __future__ import annotations

import subprocess
import sys

from common import logging as log
from common.timeutil import now_utc
from ingest.config import feeds
from store.session import connect

logger = log.get("ingest.health")

DAILY_COUNTS = """
SELECT source,
       count(*) FILTER (WHERE published_at > now() - interval '24 hours') AS last_24h,
       count(*) FILTER (WHERE published_at > now() - interval '7 days')   AS last_7d,
       count(*)                                                            AS total,
       max(retrieved_at)                                                   AS last_retrieved
FROM articles_raw
GROUP BY source
ORDER BY last_24h ASC, source
"""


def report() -> list[dict]:
    with connect() as conn:
        return conn.execute(DAILY_COUNTS).fetchall()


def notify(title: str, message: str) -> None:
    """macOS notification. Best-effort; never raises into the collector loop."""
    if sys.platform != "darwin":
        return
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification {message!r} with title {title!r}'],
            check=False, capture_output=True, timeout=10,
        )
    except Exception:
        logger.debug("notification failed", exc_info=True)


def check_dead_feeds(stale_poll_hours: int = 6, quiet_days: int = 14) -> list[str]:
    """Alert on feeds that are BROKEN, not feeds whose publisher is quiet.

    These are different failures and only one is actionable:
      * broken  -> we have not successfully polled it recently (source_health
                   .last_seen_at is stale, or the source has never been seen).
                   This is the "silent feed death" the plan cares about.
      * quiet   -> we poll it fine, the company just has not issued a press
                   release. Normal for IR feeds, which go weeks between filings.

    Keying the alert on published_at (as the first cut did) fires constantly for
    healthy IR feeds and trains you to ignore the one alert that matters.
    """
    expected = {f.source for f in feeds()} | {"sec-edgar"}
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT source,
                   max(last_seen_at) AS last_poll,
                   max(last_seen_at) FILTER (WHERE article_count > 0) AS last_content
            FROM source_health GROUP BY source
            """
        ).fetchall()
    seen = {r["source"]: r for r in rows}

    broken, quiet = [], []
    for source in sorted(expected):
        row = seen.get(source)
        if row is None or row["last_poll"] is None:
            broken.append(f"{source} (never polled)")
            continue
        age_h = (now_utc() - row["last_poll"]).total_seconds() / 3600.0
        if age_h > stale_poll_hours:
            broken.append(f"{source} (last poll {age_h:.1f}h ago)")
            continue
        last_content = row["last_content"]
        if last_content is None:
            quiet.append(source)
        else:
            quiet_h = (now_utc() - last_content).total_seconds() / 3600.0
            if quiet_h > quiet_days * 24:
                quiet.append(f"{source} ({quiet_h / 24:.0f}d)")

    if broken:
        msg = "Feeds not polling: " + ", ".join(broken)
        logger.error("DEAD FEED ALERT — %s", msg)
        notify("SwingAgent: dead feed", msg)
    if quiet:
        # Informational only. Never an alert.
        logger.info("quiet sources (polling fine, no new content): %s", ", ".join(quiet))
    return broken


def print_report() -> None:
    rows = report()
    if not rows:
        print("no articles stored yet")
        return
    print(f"{'source':<20} {'24h':>6} {'7d':>7} {'total':>8}  last retrieved")
    print("-" * 72)
    for r in rows:
        last = r["last_retrieved"].strftime("%Y-%m-%d %H:%M:%SZ") if r["last_retrieved"] else "-"
        print(f"{r['source']:<20} {r['last_24h']:>6} {r['last_7d']:>7} {r['total']:>8}  {last}")
    total = sum(r["total"] for r in rows)
    print("-" * 72)
    print(f"{'TOTAL':<20} {'':>6} {'':>7} {total:>8}")
