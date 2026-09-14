"""SEC EDGAR 8-K poller. The highest-value source in the whole news layer.

8-K is the legally required disclosure channel for material events, its
timestamp is unambiguous, and it is very often the actual catalyst that
everything in Tier 2 and 3 is merely reporting on. data-sources.md D.4.

Three rules that break people, all handled here:
  1. User-Agent with a contact email or you get 403      -> common/http.sec_headers
  2. 10 req/s per IP or you get temporarily blocked      -> common/http rate limiter
  3. CIK must be zero-padded to 10 digits in the URL     -> ingest/config.cik_map
"""
from __future__ import annotations

from typing import Any

from swing.common import logging as log
from swing.common.http import sec_get
from swing.common.timeutil import parse_iso
from swing.ingest.config import edgar_config, edgar_targets
from swing.store.raw import RawArticle, bump_health, insert_many
from swing.store.session import connect

logger = log.get("ingest.edgar")

SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"
SOURCE = "sec-edgar"
# Forms kept for RELATED companies: events, not periodic reports.
RELATED_FORMS = {"8-K", "8-K/A", "6-K"}


def _recent_filings(cik: str) -> list[dict[str, Any]]:
    """Flatten the SEC's column-oriented 'recent' block into row dicts."""
    resp = sec_get(SUBMISSIONS.format(cik=cik))
    if resp.status_code != 200:
        logger.warning("submissions %s -> HTTP %s", cik, resp.status_code)
        return []
    data = resp.json()
    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return []
    keys = list(recent.keys())
    n = len(recent.get("accessionNumber", []))
    rows = [{k: recent[k][i] for k in keys if i < len(recent[k])} for i in range(n)]
    for r in rows:
        r["_company"] = data.get("name", "")
    return rows


def _to_article(cik: str, ticker: str, row: dict[str, Any]) -> RawArticle | None:
    acc = row.get("accessionNumber")
    if not acc:
        return None
    acc_nodash = acc.replace("-", "")
    doc = row.get("primaryDocument") or ""
    url = FILING_URL.format(cik_int=int(cik), acc_nodash=acc_nodash, doc=doc)

    # acceptanceDateTime is the moment SEC accepted the filing — the real
    # publication instant, and more precise than filingDate.
    raw_ts = row.get("acceptanceDateTime") or row.get("filingDate")
    if not raw_ts:
        return None
    published = parse_iso(raw_ts)

    form = row.get("form", "")
    items = (row.get("items") or "").strip()
    desc = row.get("primaryDocDescription") or ""
    # The 8-K Item number is free labeled data for the Phase 5.2 event
    # classifier, so it goes in the headline AND stays in raw as a field.
    headline_bits = [f"{ticker} {form}"]
    if items:
        headline_bits.append(f"Item {items}")
    if desc:
        headline_bits.append(desc)
    headline = " — ".join(headline_bits)

    return RawArticle(
        url=url,
        source=SOURCE,
        headline=headline,
        summary=row.get("reportDate") and f"Report date {row['reportDate']}" or None,
        published_at=published,
        raw={
            "cik": cik,
            "ticker": ticker,
            "form": form,
            "items": items,
            "accession": acc,
            "filing_date": row.get("filingDate"),
            "report_date": row.get("reportDate"),
            "primary_document": doc,
            "company": row.get("_company"),
            "source_tier": 1,
        },
    )


def _cursor() -> dict[str, str]:
    with connect() as conn:
        rows = conn.execute("SELECT cik, last_accession FROM edgar_cursor").fetchall()
    return {r["cik"]: r["last_accession"] for r in rows if r["last_accession"]}


def _save_cursor(cik: str, ticker: str, accession: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO edgar_cursor (cik, ticker, last_accession, last_seen_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (cik) DO UPDATE
              SET last_accession = EXCLUDED.last_accession,
                  last_seen_at   = EXCLUDED.last_seen_at
            """,
            (cik, ticker, accession),
        )


def poll(first_run_limit: int = 40) -> int:
    """Poll watchlist and related-company CIKs. Returns the number of new filings.

    Related companies are included because another company's 8-K (a rival's
    earnings, a customer's capex guidance) is often what moves a watchlist stock.
    """
    cfg = edgar_config()
    wanted = set(cfg.get("forms") or ["8-K"])
    cursors = _cursor()
    total = 0

    from swing.ingest.config import stocks

    watchlist = set(stocks())
    for ticker, cik in edgar_targets().items():
        # Another company's EVENTS matter (8-K; 6-K for foreign filers). Its
        # 10-Q/10-K restates the quarter its 8-K already announced, and in
        # retrieval it crowded out the stock's own news.
        forms = wanted if ticker in watchlist else wanted & RELATED_FORMS
        try:
            rows = _recent_filings(cik)
        except Exception:
            logger.exception("edgar poll failed for %s (CIK %s)", ticker, cik)
            continue

        seen = cursors.get(cik)
        batch: list[RawArticle] = []
        for row in rows:
            if row.get("form") not in forms:
                continue
            if seen and row.get("accessionNumber") == seen:
                break  # rows are newest-first; everything below is already stored
            art = _to_article(cik, ticker, row)
            if art:
                batch.append(art)
            if not seen and len(batch) >= first_run_limit:
                break  # first run: seed recent history, don't pull years

        n = insert_many(batch)
        total += n
        if rows:
            newest = next(
                (r["accessionNumber"] for r in rows if r.get("form") in wanted), None
            )
            if newest:
                _save_cursor(cik, ticker, newest)
        if n:
            logger.info("edgar %s: %d new filings", ticker, n)

    bump_health(SOURCE, total)
    return total
