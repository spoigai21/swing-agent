"""Reading a filing's press release. Fixtures are the shapes real filings take."""
from __future__ import annotations

from swing.ingest.edgar_text import compose_headline, pick_exhibit, text_lines, title_and_lead


class TestPickExhibit:
    def test_real_exhibit_names(self):
        # Marvell, Netflix/Intel (Donnelley "d...dex991"), Tesla.
        assert pick_exhibit(["mrvl-20260827.htm", "q227_8kx812026ex-991.htm"]) == \
            "q227_8kx812026ex-991.htm"
        assert pick_exhibit(["d65144d8k.htm", "d65144dex991.htm"]) == "d65144dex991.htm"
        assert pick_exhibit(["tsla-20260722.htm", "exhibit991.htm"]) == "exhibit991.htm"
        assert pick_exhibit(["a8-k.htm", "ex99-1.htm"]) == "ex99-1.htm"

    def test_not_other_exhibits_or_images(self):
        assert pick_exhibit(["ex-9910.htm", "ex101.htm", "ex991_logo.jpg", "a8-k.htm"]) is None


class TestTitleAndLead:
    def test_press_release_skips_boilerplate(self):
        lines = text_lines(
            "<p>Exhibit 99.1</p><p>Press Release</p>"
            "<p>Marvell Technology, Inc. Reports Second Quarter of Fiscal Year 2027</p>"
            "<p>Santa Clara, Calif. (August 27, 2026) - Marvell Technology, Inc. (NASDAQ: MRVL),"
            " a leader in data infrastructure semiconductor solutions, today reported financial"
            " results for the second quarter of fiscal year 2027.</p>")
        title, lead = title_and_lead(lines, from_exhibit=True)
        assert title == "Marvell Technology, Inc. Reports Second Quarter of Fiscal Year 2027"
        assert lead.startswith("Santa Clara, Calif. (August 27, 2026)")

    def test_slide_deck_title_is_cleaned_and_letter_spacing_ignored(self):
        lines = ["Q2 2026 Update 1 Exhibit 99.1",
                 ("S E R V I C E S Announced Robotaxi Coverage Cumulative Paid Robotaxi Miles"
                  " 9 State Metro Status California SF Bay Area Safety Driver Texas Austin Ramping")]
        title, lead = title_and_lead(lines, from_exhibit=True)
        assert title == "Q2 2026 Update" and lead is None

    def test_filing_without_exhibit_reads_its_first_item(self):
        lines = ["0001018724 false 0001018724 2026-09-08 2026-09-08",
                 "Check the appropriate box below if the Form 8-K filing is intended to satisfy",
                 "Item 5.02 Departure of Directors or Certain Officers; Election of Directors",
                 ("On September 8, 2026, the Board of Directors of Amazon.com, Inc. elected Kevin R."
                  " Mandia as a director of the Company."),
                 "Item 9.01 Financial Statements and Exhibits"]
        title, lead = title_and_lead(lines, from_exhibit=False)
        assert title is None and lead.startswith("On September 8, 2026, the Board")

    def test_the_exhibit_index_item_is_never_the_content(self):
        lines = ["Item 9.01 Financial Statements and Exhibits",
                 "(d) Exhibits. The following exhibit is furnished herewith as part of this report."]
        assert title_and_lead(lines, from_exhibit=False) == (None, None)


def test_headline_keeps_form_and_items_and_adds_the_title():
    assert compose_headline("MRVL 8-K — Item 2.02,9.01 — FORM 8-K",
                            "Marvell Technology, Inc. Reports Second Quarter") == \
        "MRVL 8-K — Item 2.02,9.01 — Marvell Technology, Inc. Reports Second Quarter"
    assert compose_headline("TSM 6-K", "TSMC August 2026 Revenue Report") == \
        "TSM 6-K — TSMC August 2026 Revenue Report"
    assert compose_headline("NFLX 8-K — Item 5.02 — 8-K", None) == "NFLX 8-K — Item 5.02 — 8-K"
