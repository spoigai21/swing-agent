"""Wall-clock jobs the collector runs once a day.

The collector already runs continuously with the access macOS requires; launchd
and cron cannot read ~/Desktop without Full Disk Access (CODEBASE-PLAN §8.5,
README), so the daily work lives here rather than in a crontab.

  post-close, weekdays 16:45 ET   refresh prices, detect the day's swings, explain
                                  up to DAILY_ATTRIBUTIONS new big moves, then
                                  alert on |z| >= alert_z (agent-plan.md 6.1, 6.3)
  nightly, 00:30 Pacific          a placebo batch right after the free-tier quota
                                  resets, so Gate 4's 200 cases accumulate
                                  without anyone remembering to run them (4.3)

Budget: 3 + 12 of the 20 free Gemini requests a day, leaving 5 for questions.
Normal-day questions and repeated questions spend none.

Each job records the local day it last ran under data/, and marks it BEFORE
running: a restart must not repeat a day's alerts (they have no sent-state), and
a crash mid-run must not retry every 15 minutes and burn the quota.
"""
from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from swing.common import logging as log
from swing.paths import DATA

logger = log.get("interface.schedule")

ET = ZoneInfo("America/New_York")
PT = ZoneInfo("America/Los_Angeles")
DAILY_AT_ET = time(16, 45)          # the daily bar has settled by then
PLACEBO_AT_PT = time(0, 30)         # Gemini free-tier quotas reset at midnight Pacific
DAILY_ATTRIBUTIONS = 3
NIGHTLY_PLACEBO_CASES = 12


def due(now: datetime, zone: ZoneInfo, at: time, last_run: date | None,
        weekdays_only: bool) -> date | None:
    """The local day a once-a-day job should run for now, or None."""
    local = now.astimezone(zone)
    if weekdays_only and local.weekday() >= 5:
        return None
    if local.time() < at or last_run == local.date():
        return None
    return local.date()


def _marker(name: str):
    return DATA / f"schedule_{name}.last"


def last_run(name: str) -> date | None:
    try:
        return date.fromisoformat(_marker(name).read_text().strip())
    except (OSError, ValueError):
        return None


def mark(name: str, day: date) -> None:
    _marker(name).parent.mkdir(parents=True, exist_ok=True)
    _marker(name).write_text(day.isoformat())


def daily_if_due(now: datetime | None = None) -> int:
    day = due(now or datetime.now(UTC), ET, DAILY_AT_ET, last_run("daily"), weekdays_only=True)
    if day is None:
        return 0
    mark("daily", day)
    from swing.interface import alert, batch

    res = batch.run(day, attribution_limit=DAILY_ATTRIBUTIONS,
                    attribute_since=day - timedelta(days=3))
    alerts = alert.send(days=1)
    logger.info("scheduled daily run for %s: %s; %d alert(s)", day, res.summary(), alerts)
    return 0


def placebo_if_due(now: datetime | None = None) -> int:
    day = due(now or datetime.now(UTC), PT, PLACEBO_AT_PT, last_run("placebo"),
              weekdays_only=False)
    if day is None:
        return 0
    mark("placebo", day)
    from swing.eval.placebo import cumulative, run

    result = run(n=NIGHTLY_PLACEBO_CASES)
    logger.info("scheduled placebo for %s: %s scored this run; cumulative %s",
                day, result.get("n"), cumulative())
    return 0
