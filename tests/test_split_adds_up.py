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
from datetime import UTC, date, datetime

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


class TestAQuietDayDoesNotClaimThereIsNoNews:
    """`swing why NFLX --date 2026-09-22` said "there's no company-specific news
    to find" on a day HSBC downgraded the stock. The downgrade was in the corpus
    the whole time; what was absent was an unusual MOVE.

    Those are different claims and only one of them is measured. A user checking
    the ticker on a broker app disproves the stronger one in seconds, and that
    costs more trust than the answer was worth.
    """

    def test_no_branch_claims_the_absence_of_news(self):
        import inspect

        from swing.interface import explain

        src = inspect.getsource(explain.normal_day)
        assert "no company-specific news" not in src
        assert src.count("no company-specific move to explain") >= 2
        assert "no company-specific story to explain" not in src

    def test_company_news_is_listed_when_it_exists(self, monkeypatch):
        from swing.interface import explain

        monkeypatch.setattr(explain, "notable_company_news", lambda t, d, limit=3: [
            {"published_at": datetime(2026, 9, 22, 17, 18, tzinfo=UTC),
             "source": "analyst-ratings", "url": "",
             "headline": "HSBC downgrades Netflix (NFLX) to Hold from Buy"}])
        out = explain.normal_day(Dec(ticker="NFLX", residual=-0.002, residual_z=-0.1), "XLC")
        assert "HSBC downgrades Netflix" in out
        assert "the move does not need it" in out

    def test_nothing_is_added_when_there_is_no_news(self, monkeypatch):
        from swing.interface import explain

        monkeypatch.setattr(explain, "notable_company_news", lambda t, d, limit=3: [])
        out = explain.normal_day(Dec(residual=-0.002, residual_z=-0.1), "XLC")
        assert "company news" not in out

    def test_a_database_failure_never_breaks_the_answer(self, monkeypatch):
        """The listing is a courtesy; the decomposition is the answer."""
        from swing.interface import explain

        def boom(*a, **k):
            raise RuntimeError("db down")

        monkeypatch.setattr(explain, "notable_company_news", boom)
        out = explain.normal_day(Dec(residual=-0.002, residual_z=-0.1), "XLC")
        assert "Why:" in out

    def test_the_sources_come_from_config_not_a_literal(self):
        """Hand-written source lists had already drifted between two modules."""
        import inspect

        from swing.interface import explain

        assert "primary_sources()" in inspect.getsource(explain.notable_company_news)


class TestDisplayNamesComeFromConfig:
    """Hand-written label dicts lived in explain.py, so adding a sector to
    watchlist.yaml printed the bare ETF ticker at users, and adding a feed to
    sources.yaml printed a title-cased id. Same drift as `primary_sources`
    having two copies."""

    def test_a_sector_label_comes_from_the_watchlist(self):
        from swing.ingest.config import sector_label

        assert sector_label("XLC") == "communication stocks"

    def test_an_unknown_sector_degrades_to_its_name_then_its_ticker(self):
        from swing.ingest.config import sector_label

        assert sector_label("ZZZZ") == "ZZZZ"

    def test_source_labels_come_from_sources_yaml(self):
        from swing.ingest.config import source_labels

        labels = source_labels()
        assert labels["sec-edgar"] == "SEC filing"
        assert labels["analyst-ratings"] == "analyst rating"

    def test_an_unlisted_ir_feed_still_reads_well(self):
        from swing.interface.explain import _source_label

        assert _source_label("someco-ir") == "company press release"

    def test_explain_no_longer_carries_its_own_tables(self):
        import inspect

        from swing.interface import explain

        src = inspect.getsource(explain)
        assert "SECTOR_LABELS = {" not in src
        assert "SOURCE_LABELS = {" not in src
