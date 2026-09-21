"""What an SEC filing actually says: its press-release title and opening paragraph.

An 8-K's headline is only its form and Item numbers ("MRVL 8-K — Item 2.02"), so
the model explained Marvell's +17% day as "filed its earnings results ...
showing strong performance", and matching filings to known causes was guesswork.
The substance is in the attached press release (Exhibit 99.1) or, for an 8-K
without one, in the text under its first Item heading.

Event filings only (8-K, 8-K/A, 6-K): a 10-Q/10-K restates a quarter its 8-K
already announced. Two SEC requests per filing, through the shared rate limiter.
"""
from __future__ import annotations

import html
import json
import re

from swing.common import logging as log
from swing.common.http import sec_get
from swing.store.session import connect

logger = log.get("ingest.edgar_text")

EVENT_FORMS = ["8-K", "8-K/A", "6-K"]
# ⚠️ Matching only "ex99_1" style names missed HALF the press releases. Issuers
# name the file themselves — q2fy27pr.htm, ttwo1q27earningsrelease.htm,
# wbd2q26earningsrelease08.htm — and some file a bare ex99 with no trailing 1
# (avgo-08022026x8kxex99.htm). Every miss fell back to the 8-K COVER PAGE, which
# is why enriched summaries said "issued a press release announcing its
# unaudited financial results" instead of the results themselves: only 41 of 595
# Item 2.02 summaries contained a single figure.
# ⚠️ The (?!\d) guard is load-bearing: without it "ex-9910.htm" (exhibit 99.10,
# a different document) matches as though it were 99.1.
_EX99 = r"(?:ex|exhibit)[-_]?99(?:[-_.]?0?1)?(?!\d)"
# _EX99 is a str, not a literal, so these need explicit + rather than implicit
# string concatenation.
_EXHIBIT = re.compile(
    _EX99                                 # ex99, ex-99_1, exhibit991, a…ex991
    + r"|(?:earnings|press)[-_]?release"  # ttwo1q27earningsrelease
    + r"|(?:^|[^a-z])pr(?=[^a-z]|$)",     # q2fy27pr.htm — "pr" as its own token
    re.IGNORECASE)
# The cover page is never the press release, whatever it is called.
_COVER = re.compile(r"(?i)(?:^|[^a-z])8-?k(?:[^a-z0-9]|$)|^[a-z]{2,6}-\d{8}\.htm")
_ITEM = re.compile(r"(?i)^item\s+(\d\.\d{2})\b")
# Cover-page boilerplate, XBRL header residue and page furniture.
_JUNK = re.compile(
    r"(?i)^(\d{10}\b|.*\bfalse\b.*\d{4}-\d{2}-\d{2}|.*registration no|check the appropriate box"
    r"|.*securities exchange act|.*commission file|washington, d\.?c|form (8-k|6-k)|current report"
    r"|report of foreign private issuer|pursuant to|exhibit\s*99|ex-99|united states$"
    r"|securities and exchange commission|signatures?$|press release$|news release$"
    r"|for immediate release|contacts?:|page \d|\(?\d+\)?$|table of contents|indicate by check mark"
    r"|date of report|\((exact name|state or other|address|registrant|commission|i\.r\.s\.))")
_LETTER_SPACED = re.compile(r"(?:\b[A-Za-z]\b\s){5,}")        # slide decks: "S E R V I C E S"
_DATELINE = re.compile(r"(?i)business wire|globe ?newswire|prnewswire|\btoday\b|announced|reported")


def pick_exhibit(names: list[str]) -> str | None:
    """The press-release exhibit among a filing's files, if any.

    Prefers a release-looking name, and never returns the cover page: falling
    back to it yields boilerplate ("...issued a press release announcing...")
    rather than the numbers, which is worse than returning None because the
    caller cannot tell enrichment failed.
    """
    docs = [n for n in names
            if n.lower().endswith((".htm", ".html", ".txt"))
            and "index" not in n.lower()
            and not _COVER.search(n)]
    # Prefer an explicit exhibit-99 name, then any release-looking one.
    for n in docs:
        if re.search(_EX99, n, re.IGNORECASE):
            return n
    return next((n for n in docs if _EXHIBIT.search(n)), None)


def text_lines(markup: str) -> list[str]:
    markup = re.sub(r"(?is)<(script|style|head|ix:header)[^>]*>.*?</\1>", " ", markup)
    markup = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|h[1-6]|li|table|td)>", "\n", markup)
    markup = html.unescape(re.sub(r"<[^>]+>", " ", markup))
    return [line for line in (re.sub(r"\s+", " ", x).strip() for x in markup.split("\n")) if line]


def title_and_lead(lines: list[str], from_exhibit: bool) -> tuple[str | None, str | None]:
    """(press-release title, opening paragraph); either may be None."""
    clean = [x for x in lines if not _JUNK.match(x) and not _LETTER_SPACED.search(x)]
    if from_exhibit:
        title = next((x for x in clean if 25 <= len(x) <= 220 and len(x.split()) >= 5), None)
        if title:
            title = re.sub(r"(?i)\s+\d*\s*exhibit\s*99\.1\s*$", "", title).strip() or None
        lead = next((x for x in clean if len(x) >= 160 and _DATELINE.search(x)), None)
        return title, lead or next((x for x in clean if len(x) >= 160), None)
    # No exhibit: the first substantive Item (never 9.01, the exhibit index).
    for i, line in enumerate(lines):
        m = _ITEM.match(line)
        if m and m.group(1) != "9.01":
            para = next((x for x in lines[i + 1:i + 12]
                         if len(x) >= 80 and not _ITEM.match(x) and not _JUNK.match(x)), None)
            if para:
                return None, para
    return None, next((x for x in clean if len(x) >= 160), None)


def compose_headline(original: str, title: str | None) -> str:
    """'MRVL 8-K — Item 2.02,9.01 — FORM 8-K' -> '... — Item 2.02,9.01 — <title>'."""
    if not title:
        return original
    parts = original.split(" — ")
    keep = [parts[0]] + [p for p in parts[1:2] if p.startswith("Item ")]
    return " — ".join(keep + [title])[:2000]


def fetch(cik: str, accession: str, primary_doc: str | None) -> tuple[str | None, str | None, str]:
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"
    listing = sec_get(f"{base}/index.json")
    listing.raise_for_status()
    names = [i["name"] for i in listing.json()["directory"]["item"]]
    exhibit = pick_exhibit(names)
    # ⚠️ Falling back to primary_doc means reading the 8-K COVER PAGE, whose lead
    # is "...issued a press release announcing its unaudited financial results" —
    # boilerplate with no numbers. 554 of 595 Item 2.02 summaries had no figure
    # in them for exactly this reason. Return nothing instead, so the row stays
    # un-enriched and is retried when the picker improves, rather than being
    # recorded as done with worthless text.
    if not exhibit:
        return None, None, ""
    doc = sec_get(f"{base}/{exhibit}")
    doc.raise_for_status()
    title, lead = title_and_lead(text_lines(doc.text), True)
    return title, lead, exhibit


def enrich_pending(limit: int | None = 40, since: str = "2024-09-01") -> int:
    """Enrich event filings not yet enriched: archive row and its article (headline,
    summary, embedding, MinHash), so retrieval, ranking and the prompt all see the
    press release. Failures are left un-marked and retried on the next pass."""
    from swing.ingest.config import thresholds
    from swing.ingest.normalize import _embed_texts, _minhash

    with connect() as conn:
        rows = conn.execute(
            "SELECT r.id, r.headline, r.summary, r.raw, a.id AS article_id "
            "FROM articles_raw r LEFT JOIN articles a ON a.raw_id = r.id "
            "WHERE r.source = 'sec-edgar' AND r.raw->>'form' = ANY(%s) "
            "AND NOT (r.raw ? 'doc_enriched') AND r.published_at >= %s "
            "ORDER BY r.published_at DESC LIMIT %s",
            (EVENT_FORMS, since, limit or 100000)).fetchall()

    num_perm = int(thresholds()["dedup"].get("minhash_num_perm", 128))
    done = 0
    for start in range(0, len(rows), 32):
        chunk, updates = rows[start:start + 32], []
        for r in chunk:
            raw = r["raw"]
            try:
                title, lead, source_file = fetch(raw["cik"], raw["accession"],
                                                 raw.get("primary_document"))
            except Exception as exc:  # noqa: BLE001 - one bad filing must not stop the pass
                logger.warning("filing text failed for %s %s: %s",
                               raw.get("ticker"), raw.get("accession"), str(exc)[:80])
                continue
            updates.append((r, compose_headline(r["headline"], title),
                            (lead or "")[:600] or r["summary"],
                            {"doc_enriched": True, "doc_file": source_file, "doc_title": title,
                             "doc_lead": lead, "original_headline": r["headline"]}))
        texts = [f"{headline} {summary or ''}".strip()
                 for r, headline, summary, _extra in updates if r["article_id"]]
        vectors = iter(_embed_texts(texts)) if texts else iter(())
        with connect() as conn, conn.transaction(), conn.cursor() as cur:
            for r, headline, summary, extra in updates:
                cur.execute("UPDATE articles_raw SET headline = %s, summary = %s, raw = raw || %s "
                            "WHERE id = %s", (headline, summary, json.dumps(extra), r["id"]))
                if r["article_id"]:
                    text = f"{headline} {summary or ''}".strip()
                    mh, group = _minhash(text, num_perm)
                    cur.execute("UPDATE articles SET headline = %s, summary = %s, embedding = %s, "
                                "minhash = %s, dup_group_id = %s WHERE id = %s",
                                (headline, summary, str(next(vectors)), mh, group, r["article_id"]))
                done += 1
    if done:
        logger.info("enriched %d filings with their press-release text", done)
    return done
