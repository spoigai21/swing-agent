"""The agent grades its own homework; nothing checked the grades.

`confidence` is shown to the user and was never validated. `direction_consistent`
is trusted by `enforce_abstention` to keep a candidate alive. `magnitude_plausible`
is read by no code anywhere. None had ever been scored against the annotations.

The first run answered the question in a way no amount of reasoning would have:
both booleans were `true` in 7 of 7 candidates ever returned, so the
`direction_consistent` test in abstention has never rejected anything.
"""
from __future__ import annotations

import pytest

from swing.eval import calibration as cal


def case(confidence="high", correct=True, direction=True, magnitude=True, sid=1):
    return {"swing_id": sid, "confidence": confidence, "correct": correct,
            "direction_consistent": direction, "magnitude_plausible": magnitude}


class TestBucketMaths:
    def test_accuracy_and_interval(self):
        b = cal.Bucket("high", n=4, hits=3)
        assert b.accuracy == 0.75
        lo, hi = b.bounds()
        assert 0.0 <= lo < 0.75 < hi <= 1.0

    def test_a_perfect_small_sample_still_admits_being_bad(self):
        """3-for-3 is consistent with a true rate under 0.5; the report must
        not let a point estimate of 1.00 read as proof."""
        lo, _ = cal.Bucket("high", n=3, hits=3).bounds()
        assert lo < 0.5

    def test_an_empty_bucket_has_no_accuracy(self):
        b = cal.Bucket("low")
        assert b.accuracy is None and b.bounds() is None
        assert "no cases yet" in b.line()


class TestMonotonicity:
    """The one property that makes a confidence label worth printing."""

    def test_high_above_medium_above_low_passes(self):
        assert cal.is_monotonic([cal.Bucket("high", 10, 9), cal.Bucket("medium", 10, 6),
                                 cal.Bucket("low", 10, 2)]) is True

    def test_an_inversion_fails(self):
        assert cal.is_monotonic([cal.Bucket("high", 10, 4),
                                 cal.Bucket("medium", 10, 8)]) is False

    def test_one_bucket_is_undecidable_not_a_pass(self):
        """Returning True here would print a reassurance nothing supports."""
        assert cal.is_monotonic([cal.Bucket("high", 3, 3)]) is None

    def test_the_report_says_so_rather_than_claiming_a_pass(self, monkeypatch):
        monkeypatch.setattr(cal, "scored_cases", lambda: [case()])
        monkeypatch.setattr(cal, "flag_values", lambda f: {True: 1})
        out = cal.report()
        assert "Only one confidence level" in out

    def test_an_inverted_report_calls_the_label_decoration(self, monkeypatch):
        cases = ([case("high", correct=False, sid=i) for i in range(4)]
                 + [case("medium", correct=True, sid=10 + i) for i in range(4)])
        monkeypatch.setattr(cal, "scored_cases", lambda: cases)
        monkeypatch.setattr(cal, "flag_values", lambda f: {True: 8})
        out = cal.report()
        assert "NOT monotonic" in out and "decoration" in out


class TestAFlagWithOneValueCarriesNoInformation:
    def test_always_true_is_called_out(self, monkeypatch):
        monkeypatch.setattr(cal, "flag_values", lambda f: {True: 7})
        line = cal.flag_report("direction_consistent")
        assert "always True" in line and "carries no information" in line

    def test_a_flag_that_varies_is_not(self, monkeypatch):
        monkeypatch.setattr(cal, "flag_values", lambda f: {True: 5, False: 2})
        assert "carries no information" not in cal.flag_report("direction_consistent")

    def test_abstention_still_depends_on_that_flag(self):
        """If it is ever False the guard fires, so it must keep being checked —
        the point is that it has never been False, not that it is unused."""
        import inspect

        from swing.agent import abstention

        assert "direction_consistent" in inspect.getsource(abstention._is_valid)


class TestBucketing:
    def test_cases_land_in_their_confidence_bucket(self):
        cases = [case("high", True), case("high", False), case("medium", True)]
        got = {b.label: (b.n, b.hits) for b in cal.by_confidence(cases)}
        assert got["high"] == (2, 1) and got["medium"] == (1, 1)

    def test_an_unstated_confidence_is_not_silently_dropped(self):
        got = {b.label: b.n for b in cal.by_confidence([case("unstated")])}
        assert got.get("unstated") == 1

    def test_flag_buckets_split_true_from_false(self):
        cases = [case(direction=True, correct=True), case(direction=False, correct=False)]
        t, f = cal.by_flag("direction_consistent", cases)
        assert (t.n, t.hits) == (1, 1) and (f.n, f.hits) == (1, 0)

    def test_a_missing_flag_is_skipped_rather_than_counted_as_false(self):
        t, f = cal.by_flag("direction_consistent", [case(direction=None)])
        assert t.n == 0 and f.n == 0


class TestItMeasuresAndNeverTunes:
    def test_nothing_here_writes(self):
        """A calibration fitted to the annotations would measure nothing."""
        import inspect

        src = inspect.getsource(cal)
        for forbidden in ("INSERT", "UPDATE", "DELETE", "write_text"):
            assert forbidden not in src

    def test_no_cases_reports_instead_of_dividing_by_zero(self, monkeypatch):
        monkeypatch.setattr(cal, "scored_cases", list)
        assert "No scored cases yet" in cal.report()


class TestWiredIntoTheCli:
    def test_calibration_is_a_command(self):
        from swing.cli import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(["calibration", "--nope"])
        assert build_parser().parse_args(["calibration"]).cmd == "calibration"
