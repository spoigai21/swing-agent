#!/usr/bin/env python
"""Gate 1 verification.

1. Plot residual z-scores so every flagged spike can be eyeballed against a
   human sense of "that was a real event".
2. Print onset detail for hand-checking that onset_ts lands at the actual start
   of the move and swing_type is right. If onset is wrong, all of Phase 2 is
   wrong. agent-plan.md Gate 1.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swing.paths import DATA
from swing.store.session import connect

DEFAULT = ["NVDA", "MU", "AAPL", "TSLA", "SMH"]   # SNDK excluded: insufficient history


def plot(tickers: list[str], months: int = 6) -> str:
    start = datetime.now(UTC).date() - timedelta(days=months * 31)
    fig, axes = plt.subplots(len(tickers), 1, figsize=(13, 2.4 * len(tickers)), sharex=True)
    for ax, t in zip(axes, tickers, strict=True):
        with connect() as conn:
            rows = conn.execute(
                "SELECT d, residual_z FROM daily_factors "
                "WHERE ticker=%s AND status='ok' AND d>=%s ORDER BY d", (t, start)
            ).fetchall()
            sw = conn.execute(
                "SELECT d, residual_z, swing_type FROM swings "
                "WHERE ticker=%s AND kind='daily' AND d>=%s", (t, start)
            ).fetchall()
        ax.axhline(0, color="#999", lw=0.6)
        for lvl in (2, -2):
            ax.axhline(lvl, color="#c33", lw=0.8, ls="--")
        ax.plot([r["d"] for r in rows], [float(r["residual_z"]) for r in rows],
                lw=1.0, color="#3366cc")
        if sw:
            ax.scatter([s["d"] for s in sw], [float(s["residual_z"]) for s in sw],
                       s=26, color="#c33", zorder=3)
        ax.set_ylabel(t, rotation=0, ha="right", va="center", fontweight="bold")
        ax.grid(alpha=0.25)
    axes[0].set_title(f"Residual z-scores, last {months} months "
                      "(red = flagged swing, dashed = |z|=2)")
    fig.tight_layout()
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "gate1_residuals.png"
    fig.savefig(out, dpi=110)
    return str(out)


def onset_table(n: int = 10) -> None:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT s.ticker, s.d, s.residual_z, s.swing_type, s.onset_ts, s.volume_z,
                   s.earnings_mode, s.total_return, s.market_component, s.sector_component
            FROM swings s
            WHERE s.kind='daily' AND s.onset_source='intraday'
            ORDER BY abs(s.residual_z) DESC LIMIT %s
            """,
            (n,),
        ).fetchall()
    print(f"\n{'ticker':<7}{'date':<12}{'z':>7}{'type':>10}{'onset (ET)':>12}"
          f"{'vol_z':>7}{'earn':>6}   decomposition")
    print("-" * 108)
    for r in rows:
        et = r["onset_ts"].astimezone(ZoneInfo("America/New_York"))
        print(
            f"{r['ticker']:<7}{r['d']!s:<12}{float(r['residual_z']):>7.2f}"
            f"{r['swing_type']:>10}{et.strftime('%H:%M'):>12}"
            f"{float(r['volume_z'] or 0):>7.1f}"
            f"{'  yes' if r['earnings_mode'] else '   no':>6}"
            f"   tot {float(r['total_return']) * 100:+6.2f}% "
            f"= mkt {float(r['market_component']) * 100:+5.2f}% "
            f"+ sec {float(r['sector_component']) * 100:+5.2f}%"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=DEFAULT)
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    print("plot:", plot(a.tickers, a.months))
    onset_table(a.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
