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

Budget: up to 3 (post-close) + 17 (placebo) of the 20 free Gemini requests a
day. On a day the post-close run explains moves, that leaves none for questions
until the quota resets at midnight Pacific; normal-day and repeated questions
spend none. The placebo job stops by itself once Gate 4's 200 cases are scored
on the current version, handing the quota back.

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
DAILY_MODEL_QUOTA = 20              # free-tier Gemini, per model per day (agent/llm.py)
INTERACTIVE_RESERVE = 5             # a question YOU ask must never find the quota gone
# Gate 4 gets what is left after the daily batch and the interactive reserve.
# This was 17, which with DAILY_ATTRIBUTIONS came to exactly the 20-request day
# and starved every live question: the placebo batch fires at 00:30 PT, so the
# whole budget was spent on evaluation before breakfast and `swing` could only
# answer `model_unavailable`. Gate 4 is quota-bound either way; being asked a
# question and having nothing left is the worse failure.
NIGHTLY_PLACEBO_CASES = DAILY_MODEL_QUOTA - DAILY_ATTRIBUTIONS - INTERACTIVE_RESERVE
GATE4_CASES = 200                   # agent-plan.md 4.3


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

    scored = cumulative()["n"]
    # Size the batch against what the day ACTUALLY has left, not the constant:
    # failed calls spend quota without storing a row, so a bad night can leave
    # far less than 20. agent/llm.py counts attempts.
    from swing.agent.llm import remaining_today

    budget = max(0, remaining_today() - INTERACTIVE_RESERVE)
    if budget <= 0:
        logger.info("placebo skipped: %d requests left today, all reserved for questions",
                    remaining_today())
        return 0
    if scored >= GATE4_CASES:
        logger.info("placebo: %d cases scored on this version; Gate 4 sample complete", scored)
        return 0
    result = run(n=min(NIGHTLY_PLACEBO_CASES, GATE4_CASES - scored, budget))
    logger.info("scheduled placebo for %s: %s scored this run; cumulative %s",
                day, result.get("n"), cumulative())
    return 0
