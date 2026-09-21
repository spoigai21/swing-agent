"""Alerting on large moves. agent-plan.md 6.3.

Push on |z| >= 3, and **make the verdict prominent so `unexplained` is as
visible as an explanation.** An alert that only fires on explained moves trains
you to read the system as an explainer; the unexplained ones are the ones that
mark a coverage gap.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import date

from swing.common import logging as log
from swing.ingest.config import thresholds

logger = log.get("interface.alert")


@dataclass(slots=True)
class Alert:
    ticker: str
    d: date
    residual_z: float
    verdict: str
    headline: str

    def title(self) -> str:
        arrow = "▲" if self.residual_z > 0 else "▼"
        return f"{arrow} {self.ticker} {self.residual_z:+.1f}σ — {self.verdict.upper()}"

    def body(self) -> str:
        return f"{self.d}: {self.headline}"


def _notify(title: str, message: str) -> None:
    """macOS notification. Best-effort; never raises into a batch."""
    if sys.platform != "darwin":
        logger.info("ALERT %s — %s", title, message)
        return
    try:
        subprocess.run(
            ["osascript", "-e",
             f"display notification {message!r} with title {title!r}"],
            check=False, capture_output=True, timeout=10)
    except Exception:
        logger.debug("notification failed", exc_info=True)


def pending(days: int = 3, min_z: float | None = None) -> list[Alert]:
    """Recent large moves with a stored attribution, not yet alerted."""
    from swing.store.session import connect

    z = min_z if min_z is not None else float(thresholds()["swings"]["alert_z"])
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT s.ticker, s.d, s.residual_z, a.verdict, a.payload,
                   a.unexplained_note
            FROM attributions a JOIN swings s ON s.id = a.swing_id
            WHERE a.run_kind='production' AND abs(s.residual_z) >= %s
              AND s.d > current_date - %s::int
            ORDER BY s.d DESC, abs(s.residual_z) DESC
            """,
            (z, days),
        ).fetchall()

    out = []
    for r in rows:
        cands = (r["payload"] or {}).get("candidates") or []
        headline = (cands[0]["catalyst"] if cands
                    else (r["unexplained_note"] or "no catalyst identified"))
        out.append(Alert(r["ticker"], r["d"], float(r["residual_z"]),
                         r["verdict"], headline[:110]))
    return out


def send(days: int = 3, min_z: float | None = None, dry_run: bool = False) -> int:
    alerts = pending(days, min_z)
    for a in alerts:
        if dry_run:
            print(f"  {a.title()}\n      {a.body()}")
        else:
            _notify(a.title(), a.body())
            logger.info("alert: %s | %s", a.title(), a.body())
    return len(alerts)
