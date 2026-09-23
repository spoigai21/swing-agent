"""How much of the clock is this collector actually awake for?

News cannot be backfilled: an RSS feed serves the last 20-50 items, so an hour
the collector misses is coverage that is gone for good. That makes uptime a
DATA-QUALITY metric here, not an ops nicety — and on a laptop it is the largest
single input to catalyst coverage.

Measured on 2026-09-22 over the previous fortnight: **144 of 337 hours collected
nothing (43%)**, including one silent stretch of 101 hours. The collector log
confirms it was down rather than the feeds being quiet — 41 log lines across
those four days against 400-700 on a healthy day.

    swing uptime
    swing uptime --days 30

⚠️ An hour with no rows is not always downtime — 3am on a Sunday is genuinely
quiet. The report says which hours were silent and leaves the reading to you,
rather than declaring an outage from an absence.
"""
from __future__ import annotations

from dataclasses import dataclass

from swing.common import logging as log

logger = log.get("ingest.uptime")

#: Hours that are quiet by nature. A gap here says much less than one at 10am ET.
QUIET_HOURS_UTC = range(5, 11)          # ~1am-6am ET


@dataclass
class Hour:
    ts: object
    rows: int

    @property
    def silent(self) -> bool:
        return self.rows == 0

    @property
    def quiet_by_nature(self) -> bool:
        return self.ts.hour in QUIET_HOURS_UTC


def hourly(days: int = 14) -> list[Hour]:
    from swing.store.session import connect

    sql = """
    WITH hours AS (
      SELECT generate_series(date_trunc('hour', now() - (%s || ' days')::interval),
                             date_trunc('hour', now()), interval '1 hour') h
    ), got AS (
      SELECT date_trunc('hour', retrieved_at) h, count(*) n
      FROM articles_raw
      WHERE retrieved_at > now() - (%s || ' days')::interval
      GROUP BY 1
    )
    SELECT hours.h ts, coalesce(got.n, 0) n FROM hours LEFT JOIN got USING (h)
    ORDER BY hours.h
    """
    with connect() as conn:
        return [Hour(r["ts"], int(r["n"])) for r in conn.execute(sql, (days, days)).fetchall()]


def longest_gap(hours: list[Hour]) -> tuple[int, object | None]:
    """(length, start) of the longest run of silent hours."""
    best = run = 0
    start = best_start = None
    for h in hours:
        if h.silent:
            run += 1
            start = start or h.ts
            if run > best:
                best, best_start = run, start
        else:
            run, start = 0, None
    return best, best_start


def sparkline(hours: list[Hour]) -> str:
    """One character per hour: '#' collected, '.' silent."""
    return "".join("." if h.silent else "#" for h in hours)


def summary(days: int = 14) -> dict:
    hours = hourly(days)
    if not hours:
        return {"hours": 0}
    silent = [h for h in hours if h.silent]
    busy_silent = [h for h in silent if not h.quiet_by_nature]
    gap, gap_start = longest_gap(hours)
    return {
        "hours": len(hours),
        "silent": len(silent),
        "silent_busy": len(busy_silent),
        "uptime": 1 - len(silent) / len(hours),
        "longest_gap_h": gap,
        "longest_gap_start": gap_start,
        "rows": sum(h.rows for h in hours),
    }


def report(days: int = 14) -> str:
    hours = hourly(days)
    if not hours:
        return "\nNo collection history yet."
    s = summary(days)
    lines = [
        f"\nCollector uptime — last {days} days ({s['hours']} hours, {s['rows']:,} rows)\n",
        (f"  hours that collected something : {s['hours'] - s['silent']}  "
         f"({s['uptime']:.0%})"),
        (f"  silent hours                   : {s['silent']}"
         f"   (of which {s['silent_busy']} in normally-busy hours)"),
    ]
    if s["longest_gap_h"]:
        lines.append(f"  longest silent stretch         : {s['longest_gap_h']}h"
                     f" from {s['longest_gap_start']:%Y-%m-%d %H:%M}Z")
    lines.append("")
    # One row per day, oldest first.
    by_day: dict[object, list[Hour]] = {}
    for h in hours:
        by_day.setdefault(h.ts.date(), []).append(h)
    for day, hs in by_day.items():
        lines.append(f"  {day}  {sparkline(hs)}")
    lines += [
        "",
        "  '#' collected, '.' silent (UTC hours).",
        "",
        "  ⚠ News cannot be backfilled — RSS serves only the most recent items, so a",
        "    silent hour is coverage that is gone permanently. This is the largest",
        "    single input to catalyst coverage on a laptop.",
    ]
    return "\n".join(lines)


def gap_since_last_row() -> float | None:
    """Hours since anything was stored. Logged at startup so a restart records
    what the downtime cost, instead of the gap being invisible in hindsight."""
    from swing.store.session import connect

    with connect() as conn:
        r = conn.execute(
            "SELECT EXTRACT(epoch FROM now() - max(retrieved_at))/3600.0 h "
            "FROM articles_raw").fetchone()
    return float(r["h"]) if r and r["h"] is not None else None


def main(days: int = 14) -> int:
    print(report(days))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
