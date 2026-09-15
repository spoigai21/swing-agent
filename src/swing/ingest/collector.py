"""The Phase -1 daemon. Start it before anything else and leave it running.

Prices backfill in one call. News does not: RSS serves the last 20-50 items and
free API tiers cap historical lookback, so every hour this is not running is
Tier 3 coverage that is gone permanently. agent-plan.md, mistake #1.

Single-process scheduler: each source has its own interval and next-due time.
No external scheduler dependency, so there is nothing else to keep alive.
"""
from __future__ import annotations

import argparse
import random
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

from swing.common import logging as log
from swing.common.settings import REPO_ROOT, get_settings
from swing.ingest import analyst, edgar, health, news_finnhub, news_rss, normalize
from swing.ingest.config import edgar_config, feeds

logger = log.get("collector")
_STOP = False


@dataclass
class Job:
    name: str
    interval: float
    fn: object
    next_due: float = 0.0
    runs: int = 0
    errors: int = 0
    stored: int = 0
    jitter: float = field(default=0.1)

    def schedule(self, now: float) -> None:
        # Jitter so twelve feeds do not fire in the same instant every cycle.
        spread = self.interval * self.jitter
        self.next_due = now + self.interval + random.uniform(-spread, spread)


def build_jobs() -> list[Job]:
    from swing.ingest import edgar_text

    jobs: list[Job] = [
        Job(
            name="sec-edgar",
            interval=float(edgar_config().get("poll_seconds", 600)),
            fn=edgar.poll,
        ),
        # Read each new event filing's press release so evidence says what
        # happened, not just "8-K — Item 2.02".
        Job(name="sec-edgar-text", interval=600.0,
            fn=lambda: edgar_text.enrich_pending(limit=40)),
    ]
    for spec in feeds():
        jobs.append(
            Job(
                name=spec.id,
                interval=float(spec.poll_seconds),
                fn=(lambda s=spec: news_rss.poll_feed(s)),
            )
        )
    # Finnhub company news: one call per ticker, daily is plenty. Recommendation
    # trend changes slowly, so twice a day.
    jobs.append(Job(name="finnhub-news", interval=6 * 3600.0,
                    fn=lambda: news_finnhub.poll(days=7)))
    jobs.append(Job(name="finnhub-recs", interval=12 * 3600.0,
                    fn=news_finnhub.poll_recommendations))
    # Broker upgrades, downgrades and price targets, timestamped to the second.
    jobs.append(Job(name="analyst-ratings", interval=3600.0,
                    fn=lambda: analyst.poll(since_days=30)))

    # Drain articles_raw -> articles continuously. Without this the backlog
    # grows unbounded and nothing downstream (retrieval, clustering) sees new
    # articles at all.
    jobs.append(Job(name="normalize", interval=300.0,
                    fn=lambda: normalize.normalize_all()["written"]))

    jobs.append(
        # Every 30 min: staleness budgets are 0.7-1.0h, so a 6-hourly check
        # would let a dead feed run most of its budget before anyone looked.
        Job(name="health-check", interval=1800.0, fn=health.check_dead_feeds)
    )

    # Once-a-day work, checked every 15 min: the post-close run + alerts, and the
    # nightly placebo batch. See interface/schedule.py.
    from swing.interface import schedule

    jobs.append(Job(name="scheduled-daily", interval=900.0, fn=schedule.daily_if_due))
    jobs.append(Job(name="scheduled-placebo", interval=900.0, fn=schedule.placebo_if_due))
    return jobs


def _handle_signal(signum, _frame) -> None:
    global _STOP
    logger.info("received signal %s — finishing current job then exiting", signum)
    _STOP = True


def run(once: bool = False) -> int:
    get_settings()  # fail fast on a bad SEC_USER_AGENT rather than at 3am
    jobs = build_jobs()
    logger.info(
        "collector starting: %d jobs (%s)",
        len(jobs),
        ", ".join(f"{j.name}@{int(j.interval)}s" for j in jobs),
    )

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Stagger the first pass so the initial burst does not hit every host at once.
    now = time.monotonic()
    for i, job in enumerate(jobs):
        job.next_due = now + i * 2.0

    while not _STOP:
        now = time.monotonic()
        due = [j for j in jobs if j.next_due <= now]
        for job in due:
            try:
                result = job.fn()
                job.runs += 1
                if isinstance(result, int):
                    job.stored += result
            except Exception:
                job.errors += 1
                # A failing source must never take the collector down with it.
                logger.exception("job %s failed", job.name)
            finally:
                job.schedule(time.monotonic())

        if once and all(j.runs > 0 for j in jobs if j.name != "health-check"):
            logger.info(
                "single pass complete: %d articles stored",
                sum(j.stored for j in jobs),
            )
            return 0

        time.sleep(1.0)

    logger.info(
        "collector stopped. totals: %s",
        ", ".join(f"{j.name}={j.stored}" for j in jobs if j.stored),
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="SwingAgent news collector (Phase -1)")
    ap.add_argument("--once", action="store_true", help="run one pass and exit")
    ap.add_argument("--report", action="store_true", help="print per-source counts and exit")
    ap.add_argument("--logfile", default=str(REPO_ROOT / "data" / "collector.log"))
    args = ap.parse_args()

    log.setup(logfile=Path(args.logfile))
    if args.report:
        health.print_report()
        return 0
    return run(once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
