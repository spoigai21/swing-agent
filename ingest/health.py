"""Per-source daily counts and the dead-feed alert.

A silently dead feed for three weeks is three weeks of unrecoverable data, and
it does not announce itself. agent-plan.md Step -1.3.
"""
from __future__ import annotations

import subprocess
import sys

from common import logging as log
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


def check_dead_feeds(min_age_hours: int = 24) -> list[str]:
    """Sources with zero articles in the window. Alerts once per call."""
    rows = {r["source"]: r for r in report()}
    expected = {f.source for f in feeds()} | {"sec-edgar"}
    dead = []
    for source in sorted(expected):
        row = rows.get(source)
        if row is None or row["last_24h"] == 0:
            dead.append(source)
    if dead:
        msg = f"No articles in {min_age_hours}h from: {', '.join(dead)}"
        logger.error("DEAD FEED ALERT — %s", msg)
        notify("SwingAgent: dead feed", msg)
    return dead


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
