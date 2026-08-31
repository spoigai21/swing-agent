#!/usr/bin/env python
"""Rebuild the ticker -> CIK map from SEC's authoritative file.

Never hardcode CIKs from memory or a blog post. data-sources.md D.3.
Prints YAML you can paste into config/watchlist.yaml, and verifies the values
already there.
"""
from __future__ import annotations

import httpx

from swing.common.settings import get_settings
from swing.ingest.config import stocks

URL = "https://www.sec.gov/files/company_tickers.json"


def main() -> int:
    headers = {"User-Agent": get_settings().sec_user_agent}
    data = httpx.get(URL, headers=headers, timeout=30).json()
    lookup = {
        v["ticker"].upper(): (str(v["cik_str"]).zfill(10), v["title"])
        for v in data.values()
    }

    mismatches = 0
    for ticker, meta in stocks().items():
        found = lookup.get(ticker.upper())
        if not found:
            print(f"  {ticker:<6} NOT FOUND in SEC file")
            mismatches += 1
            continue
        cik, title = found
        configured = str(meta.get("cik", ""))
        flag = "ok" if configured == cik else f"MISMATCH (config has {configured!r})"
        if configured != cik:
            mismatches += 1
        print(f"  {ticker:<6} {cik}  {title[:38]:<38} {flag}")
    print(f"\n{mismatches} problem(s)")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
