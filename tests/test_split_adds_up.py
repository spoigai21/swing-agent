"""A split that does not add up makes a correct answer look wrong.

NFLX on 2026-09-21 printed, for a +2.2% day:

    Split: market +0.4% · communication stocks +3.2% · NFLX on its own -1.1%

Every figure was right to four decimals. The regression intercept — the stock's
average daily drift — was simply never shown, so the visible terms summed to
+3.6% against a +2.2% headline. For a tool that asks people to check its work,
that is a correctness bug in the output even though the maths was fine.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date

from swing.interface import explain


@dataclass
class Dec:
    ticker: str = "NFLX"
    d: date = date(2026, 9, 21)
    total_return: float = 0.0216
    market_component: float = 0.0036
    sector_component: float = 0.0319
    residual: float = -0.0113
    residual_z: float = -0.5572
    beta_mkt: float = 0.23
    beta_sector: float = 1.22
    r_squared: float = 0.28
    volume_z: float = 0.0
    status: str = "ok"


def shown_parts(line: str) -> list[float]:
    body = line.split("Split: ", 1)[1]
    return [float(p.rsplit(" ", 1)[1].rstrip("%")) for p in body.split(" · ")]


class TestTheSplitAlwaysAddsUp:
    def test_the_netflix_case_that_prompted_this(self):
        line = explain.header(Dec(), "XLC")
        assert "rose 2.2%" in line
        assert "its usual drift -0.3%" in line
        assert round(sum(shown_parts(line)), 1) == 2.2

    def test_the_hidden_term_is_the_regression_intercept(self):
        """Derived from the other three, so no schema or loader change."""
        terms = dict(explain.split_terms(Dec(), "XLC"))
        assert round(terms["its usual drift"], 6) == round(
            0.0216 - 0.0036 - 0.0319 - (-0.0113), 6)

    def test_a_negligible_drift_is_not_shown(self):
        """Under 0.05% it rounds to +0.0% and the rest already add up."""
        dec = Dec(total_return=0.0216, market_component=0.0036,
                  sector_component=0.0180, residual=0.0000)
        line = explain.header(dec, "XLC")
        assert "drift" not in line
        assert round(sum(shown_parts(line)), 1) == 2.2

    def test_it_holds_for_a_stock_with_no_sector(self):
        line = explain.header(Dec(sector_component=0.0), None)
        assert round(sum(shown_parts(line)), 1) == 2.2

    def test_it_holds_across_a_thousand_random_decompositions(self):
        """Rounding four terms to 0.1% can miss the total by 0.1 on its own."""
        rng = random.Random(0)
        for _ in range(1000):
            mkt = rng.uniform(-0.05, 0.05)
            sec = rng.uniform(-0.05, 0.05)
            res = rng.uniform(-0.08, 0.08)
            drift = rng.uniform(-0.01, 0.01)
            dec = Dec(total_return=mkt + sec + res + drift,
                      market_component=mkt, sector_component=sec, residual=res)
            line = explain.header(dec, "XLC")
            if "was flat" in line:        # under 0.05%: no headline figure
                assert abs(round(sum(shown_parts(line)), 1)) <= 0.1, line
                continue
            headline = float(line.split("% on")[0].split()[-1])
            assert abs(round(sum(shown_parts(line)), 1)) == headline, line

    def test_no_part_is_nudged_by_more_than_one_display_step(self):
        """Forcing the sum must never misstate a component by more than the
        0.1% the display can show. A first attempt that dumped the whole
        leftover on one term moved it by 0.14%."""
        rng = random.Random(1)
        for _ in range(500):
            vals = [rng.uniform(-0.05, 0.05) for _ in range(4)]
            shown = explain.balanced_pcts(vals, sum(vals))
            assert round(sum(shown), 1) == round(sum(vals) * 100, 1)
            for v, s in zip(vals, shown):
                assert abs(v * 100 - s) <= 0.1 + 1e-9, (v * 100, s)
