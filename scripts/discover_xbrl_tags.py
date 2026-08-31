#!/usr/bin/env python
"""Discover which XBRL revenue tags each company actually files.

Companies do not use identical tags: some file `Revenues`, others
`RevenueFromContractWithCustomerExcludingAssessedTax`, others a custom
extension. Assuming one tag works across all twelve means revenue extraction
silently works for four tickers and returns nothing for the rest.
data-sources.md D.4 / agent-plan.md TROUBLESHOOTING.

One-time, ~30 minutes for twelve companies. Write the results into
config/watchlist.yaml per ticker.
"""
from __future__ import annotations

import argparse

from swing.common.http import sec_get
from swing.ingest.config import cik_map

FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Tags that contain "Revenue" but are NOT the top line. Ranking by raw
# datapoint count picks several of these -- CostOfRevenue outranks the real
# revenue tag for GOOGL and ties it for NVDA.
EXCLUDE = (
    "costof", "deferred", "contractwithcustomerliability", "proforma",
    "increasedecrease", "unbilled", "receivable", "remainingperformance",
    "capitalized", "percentage", "concentrationrisk",
)

# SalesRevenueNet is the pre-ASC606 tag, retired around 2018. It still carries
# the most datapoints for MU/AAPL/TTWO purely because of history, so ranking by
# count silently selects a tag with no recent data. Rank by RECENCY first.
DEPRECATED = ("salesrevenuenet", "salesrevenuegoodsnet", "salesrevenueservicesnet")


def _latest_end(body: dict) -> str:
    ends = [pt.get("end", "") for unit in body.get("units", {}).values() for pt in unit]
    return max(ends) if ends else ""


def discover(cik: str, keyword: str = "Revenue", top: int = 3) -> list[tuple[str, int, str]]:
    facts = sec_get(FACTS.format(cik=cik)).json()["facts"]
    ranked: list[tuple[str, int, str]] = []
    for taxonomy in ("us-gaap", "ifrs-full"):
        for tag, body in facts.get(taxonomy, {}).items():
            low = tag.lower()
            if keyword.lower() not in low or any(x in low for x in EXCLUDE):
                continue
            n = sum(len(v) for v in body.get("units", {}).values())
            ranked.append((f"{taxonomy}:{tag}", n, _latest_end(body)))
    # Recency first, then volume: the tag the company files TODAY is the one to
    # code against. Deprecated tags sort last regardless.
    ranked.sort(key=lambda t: (t[0].split(":")[1].lower() in DEPRECATED, [-ord(c) for c in t[2]], -t[1]))
    return ranked[:top]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyword", default="Revenue")
    ap.add_argument("--ticker", help="just one ticker")
    args = ap.parse_args()

    targets = cik_map()
    if args.ticker:
        targets = {args.ticker.upper(): targets[args.ticker.upper()]}

    for ticker, cik in targets.items():
        try:
            top = discover(cik, args.keyword)
        except Exception as e:  # noqa: BLE001 - one bad CIK must not stop the sweep
            print(f"{ticker:<6} ERROR {type(e).__name__}: {e}")
            continue
        if not top:
            print(f"{ticker:<6} NO REVENUE TAG FOUND")
            continue
        best = top[0]
        print(f"{ticker:<6} -> {best[0]}   (latest {best[2]}, {best[1]} pts)")
        for tag, n, end in top[1:]:
            print(f"         alt: {tag}  (latest {end}, {n} pts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
