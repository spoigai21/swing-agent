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
# ⚠️ GATE 4 PUSH (2026-09-16). Placebo runs LATE, not at 00:30, so the day's
# questions are served first and Gate 4 sweeps up whatever quota is left.
# At 00:30 with no reserve it would take all 20 before you were awake.
# Restore normal operation: time(0, 30), DAILY_ATTRIBUTIONS 3,
# INTERACTIVE_RESERVE 5, NIGHTLY_PLACEBO_CASES = quota - batch - reserve.
PLACEBO_AT_PT = time(21, 0)         # after the close; quota resets at midnight PT
# Abstention precision is the only gate metric that has never produced a number,
# and it needs just 13 calls. Run it at 00:05 PT, right after the quota resets,
# so it gets first claim before the 21:00 placebo batch takes the day. It is
# self-terminating: once those 13 swings have a production attribution,
# `pending()` is empty and this becomes a no-op forever.
ABSTENTION_AT_PT = time(0, 5)
# Attribution accuracy is the metric that says the answers are RIGHT, and it is
# the thinnest: 3-of-3 is consistent with a true rate of 29%, under its 0.70
# target. Runs after abstention so the two eval metrics get the fresh quota
# before the 21:00 placebo batch, and self-terminates the same way.
ACCURACY_AT_PT = time(0, 20)
DAILY_ATTRIBUTIONS = 0              # push: alerts still send and cost no quota
DAILY_MODEL_QUOTA = 20              # free-tier Gemini, per model per day (agent/llm.py)
INTERACTIVE_RESERVE = 0             # push: the 21:00 slot is the protection
# Gate 4 gets what is left after the daily batch and the interactive reserve.
# This was 17, which with DAILY_ATTRIBUTIONS came to exactly the 20-request day
# and starved every live question: the placebo batch fires at 00:30 PT, so the
# whole budget was spent on evaluation before breakfast and `swing` could only
# answer `model_unavailable`. Gate 4 is quota-bound either way; being asked a
# question and having nothing left is the worse failure.
NIGHTLY_PLACEBO_CASES = DAILY_MODEL_QUOTA   # push: take everything still unspent
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


# ⚠️ These two jobs must NOT use a once-a-day marker. On 2026-09-20 both marked
# the day done before running, then attributed zero because the model returned
# 503 "high demand" and read timeouts — sidelining them for the rest of a day
# that still had 10 requests left. `pending()` is the natural terminator: when
# the swings are attributed there is nothing to do. A minimum gap between
# attempts stops a bad hour from spinning through the quota.
RETRY_GAP = timedelta(minutes=60)


def _attempt_path(name: str):
    return DATA / f".last_attempt_{name}"


def _may_attempt(name: str, now: datetime) -> bool:
    # ⚠️ The timestamp is written INTO the file, not taken from its mtime. These
    # functions are driven by a passed-in `now`, and mtime is wall-clock, so the
    # two disagree wherever the caller supplies a time — which is every test.
    path = _attempt_path(name)
    try:
        last = datetime.fromisoformat(path.read_text().strip())
    except (OSError, ValueError):
        return True
    return (now - last) >= RETRY_GAP


def _record_attempt(name: str, now: datetime) -> None:
    path = _attempt_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(now.isoformat())


def _eval_job_due(name: str, at, now: datetime | None, pending, run) -> int:
    """Run an eval backlog job: after `at`, at most hourly, until nothing pends."""
    now = now or datetime.now(UTC)
    if now.astimezone(PT).time() < at:
        return 0
    if not pending():
        return 0                      # finished; nothing left to spend quota on
    if not _may_attempt(name, now):
        return 0
    _record_attempt(name, now)
    result = run()
    logger.info("scheduled %s: %s", name, result)
    return 0


def abstention_if_due(now: datetime | None = None) -> int:
    """Attribute the annotated no-catalyst swings until none are left.

    These are swings a human researched and concluded had no findable cause, so
    `unexplained` is the right answer and a failure to abstain is exactly what
    `harness.abstention_precision` exists to catch. Gate 4 cannot see this
    failure: its placebo cases are shown DONOR evidence, whereas these are real
    swings with their own real (uninformative) news.
    """
    from swing.eval.abstention import pending, run

    return _eval_job_due("abstention", ABSTENTION_AT_PT, now, pending, run)


def accuracy_if_due(now: datetime | None = None) -> int:
    """Attribute annotated swings that have a known catalyst, until none remain.

    Gate 4 measures honesty; this measures correctness. Its bar (>0.70) needs
    n=11 to be established at 95% confidence and sits at n=3, while Gate 4's bar
    already passes at n=34 — so these requests buy more than the 35th placebo.
    """
    from swing.eval.accuracy import pending, run

    return _eval_job_due("accuracy", ACCURACY_AT_PT, now, pending, run)


def placebo_if_due(now: datetime | None = None) -> int:
    day = due(now or datetime.now(UTC), PT, PLACEBO_AT_PT, last_run("placebo"),
              weekdays_only=False)
    if day is None:
        return 0
    mark("placebo", day)
    from swing.eval.placebo import cumulative, run

    scored = cumulative()["n"]
    # ⚠️ The ledger is ADVISORY, never a gate. It has been wrong in both
    # directions: it over-counted refused requests (reading 23 on a 20-request
    # day), and after a hand-correction it under-counted (reading 7 when the
    # daily cap was already reached, because a call can be served and charged
    # without persisting an attribution). Skipping a night on a bad estimate
    # costs 20 cases; attempting when the quota is gone costs two refused calls
    # that are not charged, and placebo.run stops after two in a row.
    # Google's 429 is the only authority on what is left.
    from swing.agent.llm import remaining_today

    logger.info("placebo starting: ledger says %d requests left today (advisory)",
                remaining_today())
    if scored >= GATE4_CASES:
        logger.info("placebo: %d cases scored on this version; Gate 4 sample complete", scored)
        return 0
    result = run(n=min(NIGHTLY_PLACEBO_CASES, GATE4_CASES - scored))
    logger.info("scheduled placebo for %s: %s scored this run; cumulative %s",
                day, result.get("n"), cumulative())
    return 0
