"""GDELT: Reuters, Bloomberg, WSJ and FT headlines, timestamped to 15 minutes.

The gap Gate 2 measured is media-driven moves — a WSJ scoop, a Bloomberg report —
whose outlets no free API carries. GDELT indexes them: headline, publisher domain,
URL and the time it saw the article, never the body. Headline plus timestamp is
what the pre/post-move split needs (data-sources.md C.1), and a body was never
used for ranking anyway (agent-plan.md 0.6).

Free through BigQuery's public dataset with a service account
(GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_CLOUD_PROJECT). Cost is the bytes a query
scans: the table is partitioned by day, so one window over the configured
domains scans ~0.2 GB of the 1 TB monthly free allowance. Every query here is
bounded by `_PARTITIONTIME`; without that filter a single query would scan the
whole table.

⚠️ `DATE` is when GDELT SAW the article, within 15 minutes of publication — close
enough to place a story before or after a move, but not to the second like an
EDGAR acceptance time. Treat it as the article's timestamp, not as proof of a
minute-level ordering.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime, timedelta

from swing.common import logging as log
from swing.common.settings import REPO_ROOT
from swing.ingest.config import related_companies, sources, stocks
from swing.store.raw import RawArticle, bump_health, insert_many

logger = log.get("ingest.gdelt")
SOURCE = "gdelt"
_PAGE_TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.DOTALL)
# GDELT wraps titles in CDATA and HTML-escapes them.
_CDATA = re.compile(r"^\s*<!\[CDATA\[(.*?)\]\]>\s*$", re.DOTALL)


def config() -> dict:
    return sources().get("gdelt") or {}


def domains() -> dict[str, str]:
    """domain -> publisher name, which normalize.py tiers on."""
    return dict(config().get("domains") or {})


# ---------------------------------------------------------------- cost control
# BigQuery bills by bytes SCANNED, not rows returned, and the free allowance is
# 1 TB/month. One window over this dataset scans ~0.4 GB, so a careless loop
# spends real money: polling 12 tickers separately every hour would bill ~190
# GB/day. Two rules keep that from ever happening silently:
#   1. one query per PASS, not per ticker (the org filter is free — a 12-ticker
#      query dry-runs at the same 0.40 GB as a 1-ticker query);
#   2. a hard monthly budget, checked before every query and recorded after.
USAGE = REPO_ROOT / "data" / "gdelt_usage.json"


def budget_gb() -> float:
    """Hard monthly ceiling in GB. Below BigQuery's 1 TB free tier on purpose."""
    return float(config().get("monthly_budget_gb", 700))


def _usage() -> dict:
    try:
        return json.loads(USAGE.read_text())
    except (OSError, ValueError):
        return {}


def spent_gb(month: str | None = None) -> float:
    """GB scanned so far this calendar month."""
    month = month or datetime.now(UTC).strftime("%Y-%m")
    return float(_usage().get(month, 0.0))


def record(gb: float, month: str | None = None) -> float:
    """Add a query's scanned bytes to the month's running total."""
    month = month or datetime.now(UTC).strftime("%Y-%m")
    data = _usage()
    data[month] = round(float(data.get(month, 0.0)) + gb, 3)
    USAGE.parent.mkdir(parents=True, exist_ok=True)
    USAGE.write_text(json.dumps(data, indent=2, sort_keys=True))
    return data[month]


def affordable() -> bool:
    spent = spent_gb()
    if spent < budget_gb():
        return True
    logger.warning("gdelt paused: %.1f GB scanned this month, budget is %.0f GB. "
                   "Raise gdelt.monthly_budget_gb in config/sources.yaml to resume.",
                   spent, budget_gb())
    return False


def parse_ts(value) -> datetime:
    """GDELT's YYYYMMDDHHMMSS (UTC) -> aware datetime."""
    return datetime.strptime(str(value), "%Y%m%d%H%M%S").replace(tzinfo=UTC)


def page_title(extras: str | None) -> str | None:
    """The article headline, out of GDELT's XML `Extras` field."""
    import html

    m = _PAGE_TITLE.search(extras or "")
    if not m:
        return None
    text = m.group(1)
    cdata = _CDATA.match(text)
    title = html.unescape(cdata.group(1) if cdata else text)
    return re.sub(r"\s+", " ", title).strip() or None


def org_pattern(ticker: str) -> str:
    """The SQL LIKE body matching a company in GDELT's V2Organizations.

    GDELT names organisations ("Qualcomm", "Take Two Interactive"), never
    tickers, so the watchlist aliases are the query terms.

    ⚠️ The SHORTEST alias, not the longest name: GDELT writes "Qualcomm", never
    "QUALCOMM Inc", so matching on the full corporate name returned 0 rows for a
    window that holds the company's own launch story.
    """
    # Related companies carry their own aliases, as in config.company_name. Without
    # them GOOGL's pattern falls back to the literal "GOOGL", which GDELT — which
    # writes "ALPHABET INC" — never contains, so the query silently matches nothing.
    meta = stocks().get(ticker) or related_companies().get(ticker) or {}
    names = list(meta.get("aliases") or []) + [meta.get("name", ticker)]
    # "Take-Two" is written "Take Two" by GDELT; compare on letters only.
    cleaned = [re.sub(r"[^A-Za-z ]", " ", n).strip().upper() for n in names]
    return min((c for c in cleaned if c), key=len, default=ticker.upper())


def fetch_window(tickers, start: datetime, end: datetime, limit: int = 1000) -> int:
    """Store configured outlets' articles mentioning ANY of `tickers` in [start, end).

    One query for all tickers, never one per ticker. Two reasons:

    * Cost. BigQuery bills bytes scanned, and the organisation filter is free —
      a 12-ticker query dry-runs at the same 0.40 GB as a 1-ticker query. The
      per-ticker loop billed 12x for identical data.
    * Correctness. `insert_many` dedupes on URL, so under a per-ticker loop a
      story naming two watchlist companies was claimed by whichever ticker ran
      first and the other company never saw it.
    """
    from google.cloud import bigquery

    if isinstance(tickers, str):
        tickers = [tickers]
    patterns = {t: org_pattern(t) for t in tickers}
    client = _client()
    if client is None or not patterns or not affordable():
        return 0
    table = config().get("dataset", "gdelt-bq.gdeltv2.gkg_partitioned")
    sql = f"""
        SELECT DATE, SourceCommonName, DocumentIdentifier, Extras,
               UPPER(V2Organizations) AS orgs
        FROM `{table}`
        WHERE _PARTITIONTIME BETWEEN TIMESTAMP(@start_day) AND TIMESTAMP(@end_day)
          AND DATE BETWEEN @start_ts AND @end_ts
          AND SourceCommonName IN UNNEST(@domains)
          AND REGEXP_CONTAINS(UPPER(V2Organizations), @orgs)
          AND DocumentIdentifier IS NOT NULL
        ORDER BY DATE
        LIMIT @row_limit
    """
    params = [
        bigquery.ScalarQueryParameter("start_day", "STRING", start.date().isoformat()),
        bigquery.ScalarQueryParameter("end_day", "STRING", end.date().isoformat()),
        bigquery.ScalarQueryParameter("start_ts", "INT64", int(start.strftime("%Y%m%d%H%M%S"))),
        bigquery.ScalarQueryParameter("end_ts", "INT64", int(end.strftime("%Y%m%d%H%M%S"))),
        bigquery.ArrayQueryParameter("domains", "STRING", list(domains())),
        bigquery.ScalarQueryParameter("orgs", "STRING", "|".join(sorted(patterns.values()))),
        bigquery.ScalarQueryParameter("row_limit", "INT64", limit),
    ]
    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
    rows = job.result()                  # the RowIterator carries the row count
    batch = [a for row in rows if (a := _to_raw(row, patterns))]
    n = insert_many(batch)
    bump_health(SOURCE, n)
    # Record the spend on EVERY query, including the ones that returned nothing:
    # a month of empty queries costs exactly as much as a month of full ones.
    gb = (job.total_bytes_processed or 0) / 1e9
    month_total = record(gb)
    logger.info("gdelt %s %s..%s: %d new / %d returned (%.2f GB, %.1f/%.0f GB this month)",
                ",".join(sorted(patterns)) if len(patterns) < 4 else f"{len(patterns)} tickers",
                start.date(), end.date(), n, rows.total_rows, gb, month_total, budget_gb())
    return n


def _to_raw(row, patterns: dict[str, str]) -> RawArticle | None:
    title = page_title(row["Extras"])
    if not title:
        return None                     # a URL without a headline is not evidence
    orgs = row["orgs"] or ""
    matched = sorted(t for t, pattern in patterns.items() if pattern in orgs)
    if not matched:
        return None                     # regex hit some other company's name
    return RawArticle(
        url=row["DocumentIdentifier"],
        # The PUBLISHER, so normalize.py tiers it like any other source.
        source=domains().get(row["SourceCommonName"], row["SourceCommonName"]),
        headline=title,
        published_at=parse_ts(row["DATE"]),
        # `feed_tickers`, not `ticker`: normalize.resolve_tickers treats a bare
        # `ticker` as one authoritative company, which would throw away the
        # second company in a story about two of them.
        raw={"via": SOURCE, "feed_tickers": matched, "domain": row["SourceCommonName"],
             "gdelt_seen_at": str(row["DATE"]), "timestamp_precision": "15min"},
    )


def _client():
    """BigQuery client, or None when credentials are not configured."""
    import os

    from swing.common.settings import get_settings

    settings = get_settings()
    creds = getattr(settings, "google_application_credentials", None)
    project = getattr(settings, "google_cloud_project", None)
    if not creds or not project:
        logger.warning("GDELT skipped: GOOGLE_APPLICATION_CREDENTIALS / "
                       "GOOGLE_CLOUD_PROJECT not set")
        return None
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(creds))
    try:
        from google.cloud import bigquery

        return bigquery.Client(project=project)
    except Exception:
        logger.exception("BigQuery client unavailable")
        return None


def poll(days: int = 2) -> int:
    """Recent articles for every watchlist stock, in a single query."""
    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    try:
        return fetch_window(list(stocks()), start, end)
    except Exception:
        logger.exception("gdelt poll failed")
        return 0


def backfill_day(tickers, day: date) -> int:
    """Everything these tickers had on one calendar day (UTC)."""
    start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    return fetch_window(tickers, start, start.replace(hour=23, minute=59, second=59))
