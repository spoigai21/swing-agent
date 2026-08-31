# Data Sources and APIs — Specification

Companion to `swing-attribution-agent-plan.md`. This document fills in §0.1 (watchlist), §0.3 (price ingestion), and §0.5 (source tiers) with concrete APIs, endpoints, and configuration for your specific tickers.

---

# PART A — YOUR WATCHLIST

## A.1 Tickers

Extracted from the two screenshots. 12 stocks plus the market index.

| Ticker | Company | Sector ETF | Market |
|---|---|---|---|
| NVDA | NVIDIA | SMH | SPY |
| MRVL | Marvell Technology | SMH | SPY |
| MU | Micron Technology | SMH | SPY |
| SNDK | Sandisk | SMH | SPY |
| AVGO | Broadcom | SMH | SPY |
| QCOM | QUALCOMM | SMH | SPY |
| AAPL | Apple | XLK | SPY |
| GOOGL | Alphabet | XLC | SPY |
| NFLX | Netflix | XLC | SPY |
| TTWO | Take-Two Interactive | XLC | SPY |
| TSLA | Tesla | XLY | SPY |
| SBUX | Starbucks | XLY | SPY |

**Total symbols to ingest: 17** (12 stocks + SMH, XLK, XLC, XLY, SPY).

Use SPY rather than the S&P 500 index itself for the market factor. Same information, but you get a single consistent OHLCV series from the same endpoint as everything else.

## A.2 Ticker-specific warnings

These three issues will silently corrupt your Phase 1 output if you don't handle them up front.

### ⚠️ SNDK has almost no trading history, and what it has is unusable for beta

Sandisk began regular-way trading on Nasdaq on 24 February 2025 following its spin-off from Western Digital. That gives you roughly 18 months of history — barely enough to fill the 120-day beta window, with nothing left over.

Worse: the stock has moved a staggering amount since the spin-off, so its volatility regime has shifted radically over that short history. A rolling 120-day beta fitted across that period is fitting across two completely different regimes.

**Do this:**
- Exclude SNDK from your Phase 1 gate validation set.
- Add a `min_history_days: 400` check in `config/watchlist.yaml` and skip decomposition for any ticker below it, emitting `insufficient_history` rather than a bad residual.
- Revisit SNDK once you have 2 years of post-spin data.

Also note there is a pre-spin `SNDK` ticker from the old SanDisk, acquired by Western Digital in 2016. Some historical data sources will happily splice the two together. Verify your backfill starts on 2025-02-24 and not earlier.

### ⚠️ MU and SNDK are memory, not logic — SMH may be the wrong factor

SMH is dominated by AI-compute and logic names. MU and SNDK are NAND/DRAM businesses driven by memory pricing cycles, which move on their own schedule and frequently decouple from the rest of the sector.

If you regress MU on SMH, memory-cycle moves will show up as large idiosyncratic residuals and your agent will go looking for company news to explain what is actually an industry-wide pricing move.

**Do this:** Start with SMH for all six semis to keep it simple. After Gate 1, check the regression R² per ticker. If MU or SNDK come in materially lower than NVDA/AVGO/QCOM, add a second sector factor — either a memory peer basket or a broader storage/memory ETF. Verify current holdings before picking one; index composition changes.

### ⚠️ Fiscal calendars are all different — "Q3" means nothing without a date

Several companies on this list have fiscal years that don't align with the calendar year (NVDA, MU, AVGO, AAPL, QCOM, and TTWO all have off-calendar fiscal year ends). Two consequences:

1. **Never hardcode earnings months.** Pull the actual filing dates from EDGAR.
2. **Never compare "Q3" across tickers.** Use the `dei:DocumentFiscalYearFocus`, `dei:DocumentFiscalPeriodFocus`, and `dei:CurrentFiscalYearEndDate` facts from `companyfacts` to resolve every fiscal label to a real date range before storing it.

This also means your earnings-date suppression list from §1.3 must be built per-ticker from actual filing history, not from a calendar assumption.

---

# PART B — API LAYER 1: PRICES

## B.1 Recommended: Finnhub (live) + yfinance (backfill)

| Use | Provider | Cost | Notes |
|---|---|---|---|
| Historical backfill (2 years) | `yfinance` | Free, no key | One-time job. Pin the version. |
| Daily/live updates | Finnhub | Free tier: 60 calls/min, websockets included | `FINNHUB_API_KEY` |
| Clean EOD fallback | Tiingo | Free tier: 1,000 req/day | Optional, for training data quality |

## B.2 Rate limit budget

Your load is tiny. Do the math so you don't over-engineer:

| Job | Calls | Against limit |
|---|---|---|
| Daily EOD pull, 17 symbols | 17 | 60/min → done in one second |
| Historical backfill, 17 symbols × 2yr | 17 | One-time |
| Intraday polling (optional), 17 symbols every 5 min | 204/hour | Comfortable |

**Free tiers are genuinely sufficient at this scale.** Do not pay for market data until you expand well past 50 tickers or need sub-minute latency.

Still wrap every call in `tenacity` retry with exponential backoff — free tiers throttle unpredictably, and a silent failure that leaves a gap in your bar history will corrupt beta estimates weeks later.

## B.3 Corporate actions

Pull splits and dividends separately into the `corporate_actions` table from §0.2 of the main plan. For this watchlist specifically, verify the SNDK spin-off distribution is recorded correctly if you ingest any WDC history.

---

# PART C — API LAYER 2: NEWS

## C.1 Read this first: you cannot get WSJ full text

You asked for Wall Street Journal specifically. Here is the honest situation.

**WSJ, FT, Barron's, and Dow Jones Newswires full text are not available through any affordable API.** They are licensed through enterprise channels — Dow Jones/Factiva directly, or institutional aggregators like RavenPack/Bigdata.com which bundle premium newswires (Dow Jones, WSJ, Barron's, FT) alongside filings and transcripts. These are quant-fund products with quant-fund pricing. Any cheap API claiming WSJ content is either serving headlines only or is scraping, and scraping a paywall violates their terms.

**But here is why that matters less than it sounds.**

Your system needs, per article: **headline, timestamp, source, and entity tag.** It needs the body only to help the LLM assess relevance and framing. For the actual attribution logic — was there a pre-move catalyst, from what tier of source, and did outlets frame it differently — headline plus timestamp plus a one-sentence summary carries most of the signal.

**And WSJ publishes exactly that, for free, via RSS** at `feeds.content.dowjones.io`. Their Markets feed gives you headline, a one-line summary, the article URL, and a publication timestamp. That is enough to place a WSJ story in your timeline and cluster it against wire coverage.

**Design rule:** build your `articles` table so `body` is nullable, and make sure retrieval, clustering, and the agent prompt all work on headline-only records. If you architect around full text you'll be blocked on every premium source; if you architect around headline plus timestamp, you can ingest everything and treat body as a bonus.

## C.2 Recommended news stack

Three sources, layered. Start with all three — they cover different failure modes.

### 1. Finnhub company news (primary, ticker-tagged)

Already have the key from Layer 1. Gives ticker-tagged company news for US companies, which is exactly your watchlist. This is your workhorse: one call per ticker per date range.

Caveat worth knowing: Finnhub's news sentiment field is basic headline-level tagging. Ignore it — you're building your own analysis, and a weak sentiment score in your pipeline is worse than none.

### 2. Direct RSS (source breadth and tier-3 framing)

This is how you get the "various journals" requirement, including WSJ. Configure in `config/sources.yaml`:

| Source | Feed | Tier |
|---|---|---|
| WSJ Markets | `feeds.content.dowjones.io` public RSS | 3 |
| WSJ Tech | same host, tech feed | 3 |
| CNBC Markets | CNBC RSS | 3 |
| MarketWatch | MarketWatch RSS | 3 |
| Company IR feeds | Each company's investor-relations RSS | **1** |

**Do not skip the per-company IR feeds.** They are Tier 1, they carry press releases at the moment of publication, and for a 12-ticker watchlist it's twelve URLs. This is the single highest value-per-effort item in the whole news layer.

RSS gives you headline, summary, link, and `pubDate`. Parse with `feedparser`. Poll every 5–15 minutes.

### 3. Marketaux (entity-tagged breadth)

Free tier, entity-first filtering, good ticker tagging across a wider publisher set than Finnhub. Use it to catch coverage from outlets your RSS list doesn't include. This is your dedup stress test — the same wire story arriving through three different pipes is exactly what §2.2 exists to collapse.

### Not recommended for you right now

| Provider | Why not |
|---|---|
| Alpha Vantage NEWS_SENTIMENT | Free tier caps at 25 requests/day. Vanishes instantly across 12 tickers. |
| Benzinga (via Polygon) | Genuinely good trader-grade feed with analyst actions, but paid. Revisit if you outgrow the free stack. |
| RavenPack / Bigdata.com | Real WSJ/FT/Dow Jones coverage, institutional pricing. This is what you'd buy if this became a funded project. |
| NewsAPI.org | Free tier is development-only with delayed results and thin finance coverage. |

## C.3 Source tier assignment for this stack

Slots into §0.5 of the main plan:

- **Tier 1** — Company IR RSS feeds, SEC filings (Part D)
- **Tier 2** — Reuters/AP content arriving via Finnhub or Marketaux
- **Tier 3** — WSJ RSS, CNBC, MarketWatch, sector trade press
- **Tier 4** — Anything else. Exclude.

## C.4 Analyst actions are a coverage gap

Six semis on your list means analyst upgrades/downgrades and price-target changes will drive a meaningful share of your idiosyncratic moves — especially for QCOM, MRVL, and AVGO. The free stack covers these unevenly, and they often hit the tape before any article does.

Finnhub has recommendation-trend and price-target endpoints. Pull them daily and store them as **synthetic Tier 2 articles** with the action as the headline. Otherwise you'll get a cluster of `unexplained` verdicts that all turn out to be broker actions.

---

# PART D — API LAYER 3: FINANCIAL REPORTS (SEC EDGAR)

This is the easiest and best layer. Free, no API key, authoritative, real-time.

## D.1 Endpoints

All on `data.sec.gov`, all free, all no-key.

| Endpoint | URL pattern | Use |
|---|---|---|
| Submissions | `https://data.sec.gov/submissions/CIK##########.json` | Filing history: every form type, date, accession number. **Your 8-K monitor.** |
| Company facts | `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json` | Every XBRL fact the company has ever filed, one call. Heavy. |
| Company concept | `https://data.sec.gov/api/xbrl/companyconcept/CIK##########/{taxonomy}/{tag}.json` | One metric's time series. Cleaner when you know the tag. |
| Frames | `https://data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/{unit}/CY{year}{quarter}.json` | One concept across all filers for a period. Cross-sectional comparisons. |
| Full-text search | `https://efts.sec.gov/LATEST/search-index?q=...` | Search filing text since 2001, filter by form and date. |

## D.2 The three rules that will break you

**1. `User-Agent` header is mandatory.** Send a descriptive string with a contact email. Without it you get 403. With a vague one you risk being blocked rather than contacted.

```python
HEADERS = {"User-Agent": "SwingAgent research your.email@example.com"}
```

**2. Rate limit is 10 requests/second per IP.** Exceeding it returns 429 and can earn a temporary IP block. Add a 100ms delay between calls. Cache aggressively — company facts change quarterly at most.

**3. CIK must be zero-padded to 10 digits in the URL.** Apple's CIK is 320193, so the URL needs `CIK0000320193`. This is the single most common EDGAR bug.

## D.3 Getting your CIKs

Download `https://www.sec.gov/files/company_tickers.json` once, match your 12 tickers case-insensitively, zero-pad, and cache the mapping in `config/watchlist.yaml`. Do not hardcode CIKs from memory or from a blog post — verify from that file.

```python
import json, httpx

def build_cik_map(tickers, headers):
    r = httpx.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
    data = r.json()
    lookup = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}
    return {t: lookup[t.upper()] for t in tickers if t.upper() in lookup}
```

## D.4 What to actually monitor

### The 8-K poller — build this first

8-K is the legally required disclosure channel for material events. It is very often the *actual* catalyst that everything in your news layer is merely reporting on, and its timestamp is unambiguous.

Poll `submissions/CIK##########.json` for your 12 CIKs every 10 minutes. Filter `recentacceptanceDateTime`, diff against what you've already stored, and ingest any new 8-K as a **Tier 1 article** with `published_at` set to the SEC acceptance timestamp.

Submissions data updates with a typical processing delay under a second, and the XBRL endpoints under a minute. This is faster than most news APIs will report the same event.

Also capture the 8-K **Item number** — it tells you the event category directly (Item 2.02 is results of operations, 5.02 is officer departures, 1.01 is a material agreement). This is free labeled data for your Phase 5.2 event classifier, so store it as a field rather than burying it in the body text.

### Quarterly financials

On each new 10-Q or 10-K, pull `companyfacts` for that CIK and extract the metrics you care about. Cache the response — it's a large payload and it changes at most quarterly.

**Tag warning:** companies don't use identical XBRL tags. Some report `Revenues`, others `RevenueFromContractWithCustomerExcludingAssessedTax`. Inspect each company's `companyfacts` JSON once, record which tags they actually use in `config/watchlist.yaml`, and code against that per-ticker mapping rather than assuming a single tag works across all twelve.

### Earnings call transcripts — a real gap

EDGAR does not reliably carry transcripts. Companies sometimes file them as 8-K exhibits, but inconsistently. Finnhub offers transcripts on paid tiers. For now, treat transcripts as out of scope and rely on the 8-K Item 2.02 earnings release, which carries the numbers and the guidance language that actually move the stock.

## D.5 Bulk backfill

For historical work, `https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip` is roughly a gigabyte containing one JSON per CIK for every filer. Downloading it once beats thousands of individual API calls. Use this for your Phase 0 backfill, then switch to the live API for ongoing monitoring.

---

# PART E — CONSOLIDATED SETUP

## E.1 Environment variables

```bash
FINNHUB_API_KEY=          # prices + company news
MARKETAUX_API_KEY=        # news breadth
SEC_USER_AGENT="SwingAgent research your.email@example.com"
GEMINI_API_KEY=           # attribution LLM — Google AI Studio free tier, no card
TIINGO_API_KEY=           # optional, clean EOD
```

Note that only two of these cost anything, and one of them is your LLM provider.

## E.2 Total cost

| Layer | Cost |
|---|---|
| Prices (Finnhub free + yfinance) | $0 |
| News (Finnhub + RSS + Marketaux free) | $0 |
| Filings (SEC EDGAR) | $0 |
| LLM attribution (Gemini Flash, AI Studio free tier) | $0 |

**Total: $0.** At 12 tickers the entire system runs on free tiers end to end. Production is one to three attribution calls a day, which sits far inside the Gemini free allowance.

The only resource you actually have to ration is your Phase 4 eval sweeps — 200 placebo requests each, rerun after every prompt change. Cache the retrieval side so reruns hit only the LLM, and iterate against a 30-case smoke set before spending a full sweep.

## E.3 Build order for the data layer

1. `config/watchlist.yaml` — 12 tickers, sector ETF mapping, `min_history_days: 400`
2. CIK map from `company_tickers.json`
3. Price backfill via yfinance — 17 symbols, 2 years
4. **Verify SNDK history starts 2025-02-24 and is not spliced with pre-2016 SanDisk**
5. EDGAR 8-K poller — Tier 1, with Item numbers captured
6. Company IR RSS feeds — 12 URLs, Tier 1
7. Finnhub company news — Tier 2
8. WSJ + CNBC + MarketWatch RSS — Tier 3
9. Marketaux — breadth, exercises your dedup
10. Finnhub analyst actions as synthetic Tier 2 articles
11. Per-source daily count monitoring

Steps 1–4 clear Gate 0 for prices. Steps 5–6 alone will cover a surprising share of your real catalysts, so get them working before adding the rest.