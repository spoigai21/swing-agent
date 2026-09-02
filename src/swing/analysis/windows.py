"""swing_type -> (pre_move, post_move) retrieval windows. agent-plan.md 2.1.

NEVER merge the two buckets. Financial journalism is largely written after the
move; retrieve without this split and you build a circular attribution machine
that cites reporters reverse-engineering the same tape you are looking at.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from swing.common.timeutil import assert_utc
from swing.ingest.config import thresholds


@dataclass(slots=True)
class Window:
    start: datetime
    end: datetime

    def contains(self, ts: datetime) -> bool:
        return self.start <= assert_utc(ts) < self.end


def windows_for(swing_type: str, onset_ts: datetime, prev_close_ts: datetime,
                session_open_ts: datetime | None = None,
                drift_start_ts: datetime | None = None) -> tuple[Window, Window]:
    """Return (pre_move, post_move).

    gap      previous close -> today's open      | open -> open + 24h
    intraday onset - 24h     -> onset            | onset -> onset + 24h
    mixed    previous close  -> onset            | onset -> onset + 24h
    drift    window start - 24h -> window end    | window end -> +24h
    unknown  onset - 48h     -> onset            | onset -> onset + 24h
    """
    cfg = thresholds()["windows"]
    onset = assert_utc(onset_ts)
    prev_close = assert_utc(prev_close_ts)
    post_h = int(cfg["post_move_hours"])
    pre_h = int(cfg["intraday_pre_hours"])

    match swing_type:
        case "gap":
            open_ts = assert_utc(session_open_ts or onset)
            return (Window(prev_close, open_ts),
                    Window(open_ts, open_ts + timedelta(hours=post_h)))
        case "intraday":
            return (Window(onset - timedelta(hours=pre_h), onset),
                    Window(onset, onset + timedelta(hours=post_h)))
        case "mixed":
            return (Window(prev_close, onset),
                    Window(onset, onset + timedelta(hours=post_h)))
        case "drift":
            start = assert_utc(drift_start_ts or onset)
            buf = int(cfg["drift_pre_buffer_hours"])
            return (Window(start - timedelta(hours=buf), onset),
                    Window(onset, onset + timedelta(hours=post_h)))
        case "unknown":
            # No intraday data, so we cannot be precise. Widen and flag.
            hours = int(thresholds()["onset"]["fallback_window_hours"])
            return (Window(onset - timedelta(hours=hours), onset),
                    Window(onset, onset + timedelta(hours=post_h)))
        case _:
            raise ValueError(f"unknown swing_type: {swing_type!r}")
