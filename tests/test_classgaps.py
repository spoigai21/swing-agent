"""Phase 5.2 shipped TF-IDF because DistilBERT could not beat it, and §16.25
said the win was in labels, not architecture. Three event types have zero
examples — regulatory, litigation, management — and macro (2) and guidance (3)
are nearly as thin. macro-F1 weights every class equally, so those five are most
of what holds 5.2 back, and a classifier cannot learn a class it has never seen.

Hunting 312 unlabelled swings by hand for a litigation case is the expensive
part. This orders the queue; the annotator still decides every label.
"""
from __future__ import annotations

import re
from datetime import date

from swing.eval import classgaps as G


def row(sid, headlines, ticker="NVDA", z=-2.5, d=date(2026, 5, 1)):
    return {"id": sid, "ticker": ticker, "d": d, "residual_z": z,
            "headlines": headlines}


class TestPatterns:
    def test_every_thin_class_has_patterns(self):
        for t in ("regulatory", "litigation", "management", "macro", "guidance"):
            assert G.CLASS_PATTERNS.get(t)

    def test_they_all_compile(self):
        for patterns in G.CLASS_PATTERNS.values():
            for p in patterns:
                re.compile(p)

    def test_every_targeted_class_is_a_real_event_type(self):
        from swing.eval.annotate import EVENT_TYPES

        for t in G.CLASS_PATTERNS:
            assert t in EVENT_TYPES

    def test_it_does_not_target_classes_that_already_have_labels(self):
        """earnings (16) and product (16) do not need finding."""
        assert "earnings" not in G.CLASS_PATTERNS
        assert "product" not in G.CLASS_PATTERNS


class TestSuggesting:
    def test_only_swings_whose_coverage_matches_are_offered(self):
        rows = [row(1, ["Starbucks wins dismissal of Missouri lawsuit"]),
                row(2, ["Nvidia unveils a new GPU"])]
        got = G.suggest("litigation", rows=rows)
        assert [s.swing_id for s in got] == [1]

    def test_more_matching_headlines_rank_first(self):
        rows = [row(1, ["FTC opens probe"], z=-9.0),
                row(2, ["FTC opens probe", "DOJ antitrust suit filed"], z=-2.0)]
        assert [s.swing_id for s in G.suggest("regulatory", rows=rows)] == [2, 1]

    def test_ties_break_toward_the_bigger_move(self):
        rows = [row(1, ["lawsuit filed"], z=-2.0), row(2, ["lawsuit filed"], z=-6.0)]
        assert [s.swing_id for s in G.suggest("litigation", rows=rows)] == [2, 1]

    def test_the_matching_headline_is_shown_so_it_can_be_judged(self):
        rows = [row(1, ["Judge rules against the company in patent suit"])]
        assert "patent suit" in G.suggest("litigation", rows=rows)[0].line()

    def test_a_swing_with_no_coverage_is_skipped(self):
        assert G.suggest("macro", rows=[row(1, []), row(2, None)]) == []

    def test_the_limit_is_respected(self):
        rows = [row(i, ["Fed rate cut"]) for i in range(20)]
        assert len(G.suggest("macro", limit=3, rows=rows)) == 3


class TestItProtectsTheBlindSample:
    def test_the_report_says_to_label_these_assisted(self):
        """recall@10 is computed only over blind rows. A keyword-picked swing
        joining that set turns the metric into "recall over cases we could
        already describe"."""
        text = G.report.__doc__ or ""
        src = __import__("inspect").getsource(G.report)
        assert "ASSISTED" in src and "--blind" in src
        assert "biased" in src or "biased" in text

    def test_the_module_explains_the_same_thing_at_the_top(self):
        assert "ASSISTED MODE" in (G.__doc__ or "")

    def test_it_never_writes_a_label(self):
        import inspect

        src = inspect.getsource(G)
        for forbidden in ("INSERT", "UPDATE", "DELETE", "_save"):
            assert forbidden not in src


class TestCounts:
    def test_every_event_type_appears_even_at_zero(self, monkeypatch):
        """A class missing from the table is the one you need to see."""
        from swing.eval.annotate import EVENT_TYPES

        counts = G.label_counts()
        assert set(counts) == set(EVENT_TYPES)
