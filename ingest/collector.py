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
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import logging as log  # noqa: E402
from common.settings import REPO_ROOT, get_settings  # noqa: E402
from ingest import edgar, health, news_rss  # noqa: E402
from ingest.config import edgar_config, feeds  # noqa: E402

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
    jobs: list[Job] = [
        Job(
            name="sec-edgar",
            interval=float(edgar_config().get("poll_seconds", 600)),
            fn=edgar.poll,
        )
    ]
    for spec in feeds():
        jobs.append(
            Job(
                name=spec.id,
                interval=float(spec.poll_seconds),
                fn=(lambda s=spec: news_rss.poll_feed(s)),
            )
        )
    jobs.append(
        Job(name="health-check", interval=6 * 3600.0, fn=health.check_dead_feeds)
    )
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
