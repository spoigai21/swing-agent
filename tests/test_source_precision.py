"""Nothing learned from what the agent got right.

Every confirmed answer names the sources that produced it, and the hand-written
tier table has never been checked against those outcomes. This measures the
ratio — and refuses to act on it, because tiers drive ranking, ranking drives
which sources get cited, and citations drive this number.
"""
from __future__ import annotations

from swing.eval import sources as S


def stat(name, tier, cited, confirmed):
    return S.SourceStat(source=name, tier=tier, cited=cited, confirmed=confirmed)


class TestPrecisionMaths:
    def test_ratio(self):
        assert stat("cnbc", 3, 4, 3).precision == 0.75

    def test_an_uncited_source_has_no_precision_rather_than_zero(self):
        s = stat("barrons", 2, 0, 0)
        assert s.precision is None and "n/a" in s.line()

    def test_a_thin_sample_is_marked_unjudgeable(self):
        assert stat("cnbc", 3, 3, 3).judgeable is False
        assert stat("cnbc", 3, S.MIN_CITATIONS_TO_JUDGE, 5).judgeable is True

    def test_a_perfect_thin_sample_says_so_in_its_line(self):
        """3-for-3 at the top of a table reads as authority unless labelled."""
        assert "too few to judge" in stat("sec-edgar", 1, 3, 3).line()


class TestTierDisagreements:
    def test_a_trusted_source_that_misses_is_flagged(self):
        out = S.tier_disagreements([stat("bloomberg", 1, 20, 6)])
        assert out and "tier 1" in out[0] and "30%" in out[0]

    def test_a_distrusted_source_that_lands_is_flagged(self):
        out = S.tier_disagreements([stat("theverge", 3, 20, 19)])
        assert out and "tier 3" in out[0]

    def test_a_source_that_matches_its_tier_is_silent(self):
        assert S.tier_disagreements([stat("reuters", 1, 20, 18)]) == []

    def test_thin_samples_never_raise_a_disagreement(self):
        """The whole table is thin today; noise must not read as a finding."""
        assert S.tier_disagreements([stat("bloomberg", 1, 3, 0)]) == []


class TestOneVotePerCase:
    def test_the_counting_is_per_source_per_case(self):
        """A wire story carried by six outlets is one piece of evidence for each
        of them, not six for the syndicate."""
        import inspect

        src = inspect.getsource(S.collect)
        assert "seen" in src and "continue" in src


class TestItRefusesToCloseTheLoop:
    def test_nothing_writes_config_or_database(self):
        import inspect

        src = inspect.getsource(S)
        for forbidden in ("write_text", "yaml.dump", "INSERT", "UPDATE", "DELETE"):
            assert forbidden not in src

    def test_the_report_states_why(self, monkeypatch):
        monkeypatch.setattr(S, "collect", lambda: [stat("cnbc", 3, 12, 9)])
        assert "feedback loop" in S.report()

    def test_an_empty_corpus_reports_instead_of_dividing_by_zero(self, monkeypatch):
        monkeypatch.setattr(S, "collect", list)
        assert "No cited-and-annotated" in S.report()
