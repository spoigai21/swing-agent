"""Annotation is the bottleneck on almost every open question.

42 labels is too few to settle attribution accuracy (n=5), abstention (n=7), the
ranking-weight effect (2 of 13 held-out cases) or live-era coverage (n=4). More
labels resolve all four, and labelling costs no quota — only time. So the
research pane is where the time goes.

⚠️ The constraint that must survive any speed-up: a blind annotator must not be
shown our RANKED CLUSTER LIST, or recall@10 becomes true by construction and
measures nothing.
"""
from __future__ import annotations

import inspect

from swing.eval import annotate as A


class TestBlindnessSurvives:
    def test_journalism_is_not_offered_to_a_blind_annotator(self):
        """Primary sources say what a company or analyst SAID. Reading the day's
        coverage is encouraged — but they go find it, so the blind set keeps
        measuring what our corpus missed."""
        from swing.ingest.config import primary_sources

        for newsroom in ("bloomberg", "cnbc", "marketwatch", "wsj", "reuters", "yahoo"):
            assert newsroom not in primary_sources()

    def test_only_company_and_analyst_statements_are(self):
        from swing.ingest.config import primary_sources

        assert set(primary_sources()) == {
            "sec-edgar", "analyst-ratings", "prnewswire", "businesswire"}

    def test_the_list_lives_in_config_not_in_two_modules(self):
        """It had grown a second copy inside explain.py; a source added to one
        would have been missing from the other."""
        import inspect

        from swing.interface import explain

        assert "primary_sources()" in inspect.getsource(explain.notable_company_news)
        assert "NOTABLE_SOURCES" not in inspect.getsource(explain)

    def test_the_research_pane_is_not_our_ranked_list(self):
        src = inspect.getsource(A._research_pane)
        assert "FROM clusters" not in src
        assert "ORDER BY c.rank" not in src and "c.rank_score" not in src

    def test_clusters_are_still_revealed_only_after_committing(self):
        src = inspect.getsource(A.annotate_one)
        commit = src.index("_prompt(\"  True catalyst")
        reveal = src.index("_show_clusters")
        assert commit < reveal, "retrieval must not be visible before the answer"


class TestTheResearchPaneShowsWhatWeHold:
    def test_it_no_longer_filters_to_sec_edgar_by_raw_ticker(self):
        """That query showed 50 primary sources across 68 blind swings while the
        corpus held 436 — every IR release and analyst action was invisible."""
        src = inspect.getsource(A._research_pane)
        # The old filter survives only in the comment recording why it changed.
        query = src[src.index("filings = conn.execute"):]
        assert "source='sec-edgar' AND raw->>'ticker'" not in query
        assert "FROM articles a" in query
        assert "a.source = ANY(%s)" in query and "-ir" in query

    def test_it_shows_the_headline_not_just_a_form_number(self):
        """'8-K — Item 8.01' tells a researcher nothing on its own."""
        assert "f['headline']" in inspect.getsource(A._research_pane)


class TestEventTypeHints:
    def test_earnings_item_maps_to_earnings(self):
        assert A.event_type_hint("2.02,9.01") == "earnings"

    def test_officer_change_maps_to_management(self):
        assert A.event_type_hint("5.02") == "management"

    def test_an_acquisition_item_maps_to_m_and_a(self):
        assert A.event_type_hint("1.01,9.01") == "m_and_a"

    def test_an_unknown_item_offers_nothing_rather_than_guessing(self):
        assert A.event_type_hint("9.99") is None
        assert A.event_type_hint(None) is None
        assert A.event_type_hint("") is None

    def test_every_hint_is_a_real_event_type(self):
        for value in A.ITEM_EVENT_TYPE.values():
            assert value in A.EVENT_TYPES

    def test_the_hint_is_a_default_the_annotator_can_overrule(self):
        """Applied silently it would launder our guess into the ground truth."""
        src = inspect.getsource(A.annotate_one)
        assert "_prompt(" in src and "suggested" in src


class TestTargetedLabelling:
    def test_a_specific_swing_can_be_annotated(self):
        assert "_one_swing" in inspect.getsource(A.main)

    def test_swings_can_be_limited_to_the_live_era(self):
        """Labels from when the collector was running say what the product does
        now, rather than what it could not have known before it existed."""
        src = inspect.getsource(A._candidates)
        assert "since" in src and "s.d >= %s::date" in src

    def test_the_cli_passes_both_through(self):
        from swing.cli import build_parser

        args = build_parser().parse_args(["annotate", "--swing", "293",
                                          "--since", "2026-08-01"])
        assert args.swing == 293 and args.since == "2026-08-01"
