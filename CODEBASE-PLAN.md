# Codebase Plan — Stock Swing Attribution Agent

Derived from `agent-plan.md` + `data-sources.md`. This document is the *engineering* layer: it
resolves the gaps and conflicts between those two specs, fixes the concrete module boundaries,
completes the database schema, and defines the build order.

The source docs describe **what** to build and **why**. This describes **where each thing lives**
and **what its signature is**.

---

# 0 — What this plan resolves

The two specs are strong on rationale but leave nine things undefined that you cannot write code
without. Each is resolved below and marked ⚑ where it appears in the tree.

| # | Gap in the source docs | Resolution here |
|---|---|---|
| G1 | `swings`, `clusters`, `attributions`, `intraday_bars`, `annotations` tables are referenced but never defined (only an `ALTER TABLE attributions` appears, §3.1b) | Full DDL in §3 |
| G2 | `articles_raw` (Phase −1) and `articles` (§0.4) are two different tables with no defined path between them | Explicit `normalize` stage, §4.3 |
| G3 | `residual_history` is used for the z-score denominator (§1.2) but nothing says how it is produced | Persisted `daily_factors` table + one vectorized rolling pass, §4.4 |
| G4 | `cluster_id` is a column on `articles` (implies global clustering) but agglomerative clustering is not incremental | Two-tier: global incremental `dup_group_id` + window-scoped persisted clusters, §4.6 |
| G5 | Onset detection needs 5-min bars for the stock **and** its sector; no storage or acquisition path defined | `intraday_bars` table + capture-on-detect policy, §4.5 |
| G6 | Sector ETFs are both *factors* and *entities* (§0.1b) — no stated ordering or reuse | Single `Entity` abstraction, sector-first batch ordering, §4.4 |
| G7 | Config layout given, but `config_hash` (§3.1b) has no defined input set | `common/versioning.py`, hashes a canonicalised subset, §4.9 |
| G8 | No process model — what is a daemon, what is a cron job, what is a CLI | §5 |
| G9 | No test strategy, no migrations dir, despite `alembic` and `pytest` in requirements | §2 tree + §6 |

## 0.1 — Corrections to the source docs

Three factual claims in the specs are worth verifying on day one, because a build depends on each.

**⚠️ C1 — "Finnhub's free tier covers intraday" (agent-plan §1.3) is likely wrong now.**
Finnhub moved `/stock/candle` (OHLCV candles, incl. intraday resolutions) behind paid plans. Since
onset detection is one of the two *irrecoverable* mistakes in the plan, do not let it depend on an
unverified endpoint. **Verify this before Phase 1.** Plan of record: use `yfinance` for 5-minute
bars (`interval="5m"`, ~60 calendar days of lookback), and persist every swing day's intraday bars
permanently the moment a swing is detected — the lookback limit then only affects backfill, never
going forward. This is exactly the mitigation the plan already prescribes in TROUBLESHOOTING
("store intraday bars for every swing day as you detect it"); it just becomes the primary path
rather than the fallback.

**⚠️ C2 — Marketaux free tier is thinner than the doc implies.**
It is roughly 100 requests/day and returns ~3 articles per response. That is a breadth *sampler*,
not a feed. Keep it (it is genuinely useful as the dedup stress test the doc describes) but do not
schedule it as if it were a primary source. Priority stays: EDGAR → IR RSS → Finnhub → tier-3 RSS.

**⚠️ C3 — `ivfflat` index in §0.4's DDL contradicts the TROUBLESHOOTING section.**
The schema block creates the index at table-creation time; TROUBLESHOOTING correctly says that
produces a useless partition on an empty table. **Use `hnsw` and sidestep the failure mode
entirely** — it has no build-order dependency and is more forgiving at this scale (tens of
thousands of rows). The DDL in §3 reflects this.

## 0.2 — Environment findings (probed, not assumed)

| Requirement | Spec | This machine | Action |
|---|---|---|---|
| Python | 3.12 | 3.11.9 and 3.13.9 only | Install 3.12 (`uv python install 3.12` is fastest) |
| PostgreSQL 16 + pgvector | required | **not installed** | Docker: `pgvector/pgvector:pg16` |
| Docker | — | 28.3.2 ✅ | Use it for Postgres |
| git repo | — | **not initialised** | `git init` before Phase −1 |

Do not substitute 3.13 for 3.12. The spec's warning about wheel gaps in this dependency set
(`torch`, `lightgbm`, `psycopg[binary]`, `pyarrow`) is the correct call.

---

# 1 — Architectural shape

Five layers, strictly one-directional. Nothing below reaches upward.

```
  ┌────────────────────────────────────────────────────────┐
  │  L5  interface/     CLI, query, alerting, dashboards    │
  ├────────────────────────────────────────────────────────┤
  │  L4  agent/         LangGraph StateGraph + Gemini       │
  ├────────────────────────────────────────────────────────┤
  │  L3  analysis/      decompose, swings, onset, dedup     │
  ├────────────────────────────────────────────────────────┤
  │  L2  store/         SQLAlchemy models + typed queries   │
  ├────────────────────────────────────────────────────────┤
  │  L1  ingest/        prices, news, EDGAR, analyst        │
  └────────────────────────────────────────────────────────┘
        common/   settings, time, http, versioning, logging
                  (imported by every layer, imports none)
```

Two rules that keep this honest:

1. **`analysis/` never calls an LLM and never makes a network request.** It is pure functions over
   arrays and rows. This is what makes Phase 1 testable and what makes the Phase 4 placebo harness
   able to replay retrieval without re-fetching.
2. **`agent/` never touches the network except through the LLM client.** All data arrives as a
   fully-materialised state dict built by `store/` and `analysis/`. This is what lets the placebo
   test cache the retrieval side and hit only the LLM on re-runs (§4.3 of the spec).

---

# 2 — Repository tree

Build in-place at `stock-agent-analysis/` (not a nested `swing-agent/` subdir — the specs live here).
⚑ marks a file that is **not** in the source docs' layout and resolves a gap from §0.

```
stock-agent-analysis/
├── agent-plan.md
├── data-sources.md
├── CODEBASE-PLAN.md              ← this file
├── README.md                     ⚑ runbook: how to start/stop the collector, run a batch
├── pyproject.toml                ⚑ project metadata + tool config (ruff, pytest)
├── requirements.txt
├── requirements.lock.txt
├── .env.example                  ⚑ committed; .env is gitignored
├── .gitignore                    ⚑
├── docker-compose.yml            ⚑ postgres 16 + pgvector, the only container
│
├── config/
│   ├── watchlist.yaml            # 12 tickers + 4 sector entities; CIK, IR feed, XBRL tags, min_history_days
│   ├── sources.yaml              # source → tier, RSS URLs, poll intervals
│   ├── thresholds.yaml           # z cutoffs, windows, ranking weights, dedup thresholds
│   └── prompts/                  ⚑ versioned prompt text, one file per version
│       ├── attribution_v1.md
│       └── attribution_v2.md
│
├── common/                       ⚑ entire package — imported everywhere, imports nothing internal
│   ├── settings.py               # pydantic-settings; loads .env, validates keys present
│   ├── timeutil.py               # assert_utc(), market_session(), to_et(), trading_days()
│   ├── http.py                   # httpx client factory + tenacity policies per provider
│   ├── versioning.py             # config_hash(), prompt_version(), model_id resolution  (G7)
│   └── logging.py                # structured logging, one line per ingest/attribution event
│
├── store/
│   ├── schema.sql                # canonical DDL — the source of truth
│   ├── models.py                 # SQLAlchemy ORM mirroring schema.sql
│   ├── queries.py                # every read the rest of the system needs, typed
│   ├── session.py                ⚑ engine/session factory, pgvector registration
│   └── migrations/               ⚑ alembic (deps list it; the source layout omits it)
│       ├── env.py
│       └── versions/
│
├── ingest/
│   ├── collector.py              ⚑ THE PHASE −1 DAEMON. Scheduler loop over all pollers.
│   ├── prices.py                 # yfinance backfill + Finnhub daily EOD
│   ├── intraday.py               ⚑ 5-min bars for swing days (stock + sector)          (G5, C1)
│   ├── news_rss.py               ⚑ feedparser: IR feeds + WSJ/CNBC/MarketWatch
│   ├── news_finnhub.py           ⚑ company news (split from news.py for testability)
│   ├── news_marketaux.py         ⚑ breadth sampler, low frequency                       (C2)
│   ├── analyst.py                ⚑ Finnhub recommendation/price-target → synthetic Tier-2 articles
│   ├── edgar.py                  # 8-K poller, submissions, companyfacts, CIK map
│   ├── normalize.py              ⚑ articles_raw → articles: tier, tickers, UTC, embed   (G2)
│   └── health.py                 ⚑ per-source daily counts + dead-feed alert
│
├── analysis/
│   ├── factors.py                ⚑ rolling beta fit over full history → daily_factors   (G3)
│   ├── decompose.py              # single-day decomposition (the spec's §1.1)
│   ├── swings.py                 # z-threshold, drift detector, volume_z, earnings_mode
│   ├── onset.py                  ⚑ split out of swings.py — it is the critical path     (G5)
│   ├── windows.py                ⚑ swing_type → (pre_window, post_window)
│   ├── dedup.py                  # MinHash LSH (global) + semantic clustering (windowed) (G4)
│   ├── rank.py                   ⚑ the §2.3 scoring function, weights from config
│   └── novelty.py                # heuristic first; MLP in models/
│
├── agent/
│   ├── graph.py                  # StateGraph wiring
│   ├── state.py                  ⚑ TypedDict state schema (spec §3.3 mandates TypedDict)
│   ├── nodes.py                  ⚑ one function per node, each independently testable
│   ├── schema.py                 # Attribution / AttributionFlat + rehydrate()
│   ├── abstention.py             ⚑ enforce_abstention() — split out; most important fn in the agent
│   ├── validate.py               ⚑ validate_citations()
│   ├── llm.py                    ⚑ provider abstraction (spec §1.1: "one config line" to swap)
│   └── prompts.py                # loads config/prompts/*.md, records which version was used
│
├── models/
│   ├── train_novelty.py          # MLP vs. relevance-only baseline
│   ├── train_events.py           # DistilBERT vs. LightGBM+TF-IDF baseline
│   ├── train_retrieval.py        # bi-encoder contrastive, walk-forward splits
│   └── splits.py                 ⚑ walk-forward split helper — shared, so no file can shuffle
│
├── eval/
│   ├── annotate.py               # blind + assisted annotation CLI
│   ├── harness.py                # the five metrics
│   ├── placebo.py                # 200-case confabulation test
│   └── cache.py                  ⚑ persisted cluster payloads so re-runs hit only the LLM
│
├── interface/                    ⚑ (spec Phase 6 has no home in its own layout)
│   ├── cli.py                    # single Typer/argparse entrypoint: swing <command>
│   ├── batch.py                  # daily post-close run
│   ├── query.py                  # NL over stored attributions
│   └── alert.py                  # |z| >= 3 push
│
├── scripts/                      ⚑ one-shot operational scripts
│   ├── bootstrap_db.py
│   ├── backfill_prices.py
│   ├── discover_xbrl_tags.py     # the §TROUBLESHOOTING tag-discovery routine
│   └── build_cik_map.py
│
└── tests/                        ⚑
    ├── conftest.py               # in-memory/ephemeral pg fixture, frozen clocks
    ├── test_decompose.py         # synthetic returns with known betas
    ├── test_onset.py             # hand-built bar series: gap / intraday / mixed
    ├── test_windows.py           # boundary conditions on every swing_type
    ├── test_abstention.py        # the code-level guarantee, exhaustively
    ├── test_dedup.py
    ├── test_timeutil.py          # naive datetimes must raise
    └── fixtures/
```

**Files split out from the source layout, and why:** `onset.py` (spec's own "irrecoverable mistake
#2" — it deserves its own file and its own test module), `abstention.py` (spec calls it "the most
important function in the agent"), `factors.py` (G3), `normalize.py` (G2), `collector.py` (the
Phase −1 daemon has no home in the source tree), `llm.py` (the spec requires a swappable provider
but puts the model config inline in `graph.py`).

---

# 3 — Data model

`store/schema.sql` is the source of truth; `models.py` mirrors it. Tables marked ⚑ do not exist in
the source docs (G1).

### 3.1 Ingest tier

```sql
-- Phase −1. Write-only, never deleted. The archive of record.
CREATE TABLE articles_raw (
  id           bigserial PRIMARY KEY,
  url          text UNIQUE NOT NULL,
  source       text NOT NULL,
  headline     text NOT NULL,
  summary      text,
  body         text,
  published_at timestamptz NOT NULL,
  retrieved_at timestamptz NOT NULL DEFAULT now(),
  raw          jsonb,
  normalized   boolean NOT NULL DEFAULT false   -- ⚑ G2: normalize.py's work queue
);
CREATE INDEX ON articles_raw (normalized) WHERE normalized = false;
CREATE INDEX ON articles_raw (source, published_at);
```

### 3.2 Price tier

```sql
CREATE TABLE bars (                       -- unadjusted daily OHLCV
  ticker text NOT NULL, ts timestamptz NOT NULL,
  open numeric, high numeric, low numeric, close numeric, volume bigint,
  PRIMARY KEY (ticker, ts)
);

CREATE TABLE corporate_actions (
  ticker text NOT NULL, ex_date date NOT NULL,
  kind text NOT NULL,                     -- 'split' | 'dividend'
  ratio numeric, amount numeric,
  PRIMARY KEY (ticker, ex_date, kind)
);

-- ⚑ G5/C1. Captured permanently on swing detection; free-tier lookback then
-- only limits backfill, never forward operation.
CREATE TABLE intraday_bars (
  ticker text NOT NULL, ts timestamptz NOT NULL,
  open numeric, high numeric, low numeric, close numeric, volume bigint,
  interval_sec int NOT NULL DEFAULT 300,
  PRIMARY KEY (ticker, ts, interval_sec)
);
CREATE INDEX ON intraday_bars (ticker, ts);
```

### 3.3 Article tier

```sql
CREATE TABLE articles (
  id            bigserial PRIMARY KEY,
  raw_id        bigint REFERENCES articles_raw(id),   -- ⚑ provenance back to the archive
  url           text UNIQUE NOT NULL,
  source        text NOT NULL,
  source_tier   int  NOT NULL CHECK (source_tier BETWEEN 1 AND 4),
  headline      text NOT NULL,
  summary       text,
  body          text,                                  -- nullable by design (spec §0.6)
  published_at  timestamptz NOT NULL,
  retrieved_at  timestamptz NOT NULL,
  tickers       text[] NOT NULL,
  event_hint    text,                                  -- ⚑ 8-K Item number, e.g. '2.02'
  embedding     vector(768),                           -- headline + summary ONLY
  minhash       bytea,
  dup_group_id  bigint                                 -- ⚑ G4: global, incremental, MinHash-derived
);
CREATE INDEX ON articles (published_at);
CREATE INDEX ON articles USING gin (tickers);
CREATE INDEX ON articles (dup_group_id);
-- hnsw, not ivfflat (C3): no build-order dependency, no empty-table failure mode.
CREATE INDEX articles_embedding_idx ON articles
  USING hnsw (embedding vector_cosine_ops);
```

### 3.4 Analysis tier ⚑ (entirely absent from the source docs)

```sql
-- G3: one row per (ticker, day). Produced by one vectorized rolling pass, so the
-- z-score denominator in §1.2 is consistent, cheap, and reproducible.
CREATE TABLE daily_factors (
  ticker           text NOT NULL,
  d                date NOT NULL,
  ret              numeric NOT NULL,
  alpha            numeric, beta_mkt numeric, beta_sector numeric,
  r_squared        numeric,                  -- data-sources A.2: the MU/SMH check
  market_component numeric, sector_component numeric,
  residual         numeric,
  residual_vol_60  numeric,                  -- the z denominator, stored not recomputed
  residual_z       numeric,
  volume_z         numeric,
  status           text NOT NULL DEFAULT 'ok',  -- 'ok' | 'insufficient_history' | 'gap_in_bars'
  PRIMARY KEY (ticker, d)
);

CREATE TABLE swings (
  id            bigserial PRIMARY KEY,
  ticker        text NOT NULL,
  d             date NOT NULL,
  kind          text NOT NULL,     -- 'daily' | 'drift'
  drift_window  int,               -- 3 or 5 for drift, NULL for daily
  residual      numeric NOT NULL,
  residual_z    numeric NOT NULL,
  total_return  numeric NOT NULL,
  market_component numeric NOT NULL,
  sector_component numeric NOT NULL,
  volume_z      numeric,
  swing_type    text NOT NULL,     -- 'gap' | 'intraday' | 'mixed' | 'drift' | 'unknown'
  onset_ts      timestamptz,       -- NULL only when swing_type='unknown'
  onset_source  text,              -- ⚑ 'intraday' | 'fallback_48h' — weaker evidence, flagged
  earnings_mode boolean NOT NULL DEFAULT false,
  entity_type   text NOT NULL DEFAULT 'stock',  -- 'stock' | 'sector'   (G6)
  superseded_by bigint REFERENCES swings(id),   -- ⚑ drift/daily dedup (spec §1.5)
  created_at    timestamptz NOT NULL DEFAULT now(),
  UNIQUE (ticker, d, kind, drift_window)
);

-- G4: window-scoped semantic clusters. Persisted so an attribution's citations
-- remain resolvable forever, even after thresholds are retuned.
CREATE TABLE clusters (
  id                bigserial PRIMARY KEY,
  swing_id          bigint NOT NULL REFERENCES swings(id) ON DELETE CASCADE,
  timing            text NOT NULL,      -- 'pre_move' | 'post_move'
  canonical_article bigint NOT NULL REFERENCES articles(id),
  member_count      int NOT NULL,
  distinct_sources  int NOT NULL,       -- the corroboration count
  earliest_published timestamptz NOT NULL,
  best_tier         int NOT NULL,
  semantic_score    numeric, timing_score numeric,
  novelty_score     numeric, rank_score numeric,
  rank              int
);
CREATE INDEX ON clusters (swing_id, rank);

CREATE TABLE cluster_members (
  cluster_id bigint NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
  article_id bigint NOT NULL REFERENCES articles(id),
  PRIMARY KEY (cluster_id, article_id)
);
```

### 3.5 Output + evaluation tier ⚑

```sql
CREATE TABLE attributions (
  id             bigserial PRIMARY KEY,
  swing_id       bigint NOT NULL REFERENCES swings(id),
  verdict        text NOT NULL,          -- 'explained'|'partially_explained'|'unexplained'
  payload        jsonb NOT NULL,         -- the validated Attribution model
  unexplained_note text,
  -- spec §3.1b — populated by us, never by the model
  prompt_version text NOT NULL,
  model_id       text NOT NULL,
  config_hash    text NOT NULL,
  run_kind       text NOT NULL DEFAULT 'production',  -- ⚑ 'production'|'placebo'|'eval'
  created_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON attributions (swing_id, created_at DESC);

CREATE TABLE annotations (               -- ⚑ the asset the whole project rests on (spec §4.1)
  swing_id         bigint PRIMARY KEY REFERENCES swings(id),
  blind            boolean NOT NULL,     -- recall@10 computed ONLY over blind=true
  true_catalyst    text,
  true_cluster_id  bigint REFERENCES clusters(id),
  true_event_type  text,
  no_catalyst      boolean NOT NULL DEFAULT false,
  annotator_note   text,
  annotated_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE source_health (             -- ⚑ spec §6.4 / Phase −1.3 dead-feed alert
  source text NOT NULL, d date NOT NULL,
  article_count int NOT NULL,
  PRIMARY KEY (source, d)
);
```

---

# 4 — Module contracts

Signatures, so files can be written in parallel and tested in isolation.

### 4.1 `common/timeutil.py`
```python
def assert_utc(ts: datetime) -> datetime            # raises on naive; converts to UTC
def prev_session_close(ticker: str, d: date) -> datetime
def session_open(d: date) -> datetime               # 13:30 UTC / 14:30 UTC by DST
def trading_days(start: date, end: date) -> list[date]
```
Called on **every** write and **every** read of a timestamp. This is the guard against the
"timezone bugs that don't announce themselves" failure mode.

### 4.2 `common/settings.py`
```python
class Settings(BaseSettings):
    finnhub_api_key: str
    gemini_api_key: str
    sec_user_agent: str                 # validated to contain '@' — SEC 403 guard
    marketaux_api_key: str | None = None
    tiingo_api_key: str | None = None
    database_url: str
    attribution_model: str = "google_genai:gemini-flash-latest"
```
Fails loudly at import if a required key is missing, rather than at 3am inside a poller.

### 4.3 `ingest/normalize.py` — resolves G2
```python
def normalize_batch(limit: int = 256) -> int
    # articles_raw WHERE normalized=false
    #  → assign source_tier from config/sources.yaml (drop tier 4 entirely)
    #  → tag tickers (Finnhub tag, EDGAR CIK, or watchlist keyword match)
    #  → assert_utc on published_at
    #  → embed headline + ' ' + summary   (NEVER body — spec §0.6)
    #  → MinHash → dup_group_id
    #  → INSERT INTO articles; mark raw row normalized
```
Idempotent and re-runnable. Because the archive is never mutated, changing the tier map or the
embedding model is a full re-run of this stage, not a re-fetch.

### 4.4 `analysis/factors.py` — resolves G3, G6
```python
def fit_history(entity: Entity, bars: pd.DataFrame,
                mkt: pd.DataFrame, sector: pd.DataFrame | None,
                lookback: int = 120) -> pd.DataFrame
    # One vectorized rolling pass → one daily_factors row per day.
    # sector=None for entity_type='sector' (SPY-only regression, spec §0.1b).
    # Emits status='insufficient_history' when history < min_history_days (SNDK).
    # Asserts no NaN and complete windows before every fit (spec TROUBLESHOOTING).
```
`Entity` is one abstraction over both stocks and sector ETFs, so the sector-first batch ordering
in §0.1b is a sort key rather than a separate code path.

### 4.5 `analysis/onset.py` — resolves G5, the critical path
```python
def locate_onset(stock_bars, sector_bars, prev_close, beta_sector,
                 beta_mkt, mkt_bars) -> Onset            # (ts, swing_type, source)
def fallback_onset(d: date) -> Onset                     # swing_type='unknown', 48h window
```
Two deviations from the spec's sketch, both deliberate:
- The spec's `find_onset` neutralises only the sector. Use **both** daily-fit betas intraday
  (market and sector) so the intraday abnormal return is the same quantity the daily residual is.
- Return an explicit `source` field so `onset_source='fallback_48h'` rows can be excluded from
  ranking-weight tuning, as TROUBLESHOOTING requires.

### 4.6 `analysis/dedup.py` — resolves G4
Two tiers, because agglomerative clustering is not incremental and `articles.cluster_id` as
written implies it is:

- **Global, incremental:** MinHash LSH at normalize time → `articles.dup_group_id`. Stable, cheap,
  catches verbatim syndication. Runs once per article, forever.
- **Window-scoped, at retrieval:** agglomerative clustering over the pre/post windows of one swing
  → rows in `clusters` + `cluster_members`. Persisted, so a citation stays resolvable even after
  thresholds are retuned.

```python
def dup_group_for(article: Article) -> int
def cluster_window(articles: list[Article], swing_id: int, timing: str,
                   cosine_threshold: float) -> list[Cluster]
```

### 4.7 `agent/state.py`
```python
class AttributionState(TypedDict):
    swing: SwingRow
    decomposition: Decomposition
    pre_clusters: list[ClusterView]
    post_clusters: list[ClusterView]
    raw_output: AttributionFlat | None
    attribution: Attribution | None
    verdict_reason: str | None
    messages: Annotated[list[AnyMessage], add_messages]   # spec §3.3: the reducer is mandatory
```

### 4.8 `agent/llm.py`
```python
def get_attributor() -> Runnable
    # init_chat_model(settings.attribution_model, temperature=0)
    # thinking/reasoning budget disabled  (spec §3.4 — abstention recall)
    # .with_structured_output(AttributionFlat)   ← flat by default (spec §3.1 + TROUBLESHOOTING)
    # tenacity retry on 429 + deliberate inter-call sleep
```
**Build flat-first.** The spec says test nesting and flatten if it fails; the flat schema costs
five lines of `rehydrate()` and removes an entire class of failure. Do not discover this in Phase 4.

### 4.9 `common/versioning.py` — resolves G7
```python
def config_hash() -> str      # sha256 over canonicalised thresholds.yaml + sources.yaml
                              # + watchlist.yaml + active ranking weights
def prompt_version() -> str   # filename stem of the active config/prompts/*.md
def model_id() -> str         # the resolved model string actually used, not the alias
```
Computed at runtime from the files themselves (spec §3.1b: "not by hand").

---

# 5 — Process model — resolves G8

Four things run, on different clocks. Keeping them separate is what lets the collector stay up
while everything else is rebuilt.

| Process | Cadence | Entrypoint | Notes |
|---|---|---|---|
| **Collector daemon** | continuous | `ingest/collector.py` | EDGAR 10 min · IR RSS 10 min · tier-3 RSS 15 min · Finnhub news daily · Marketaux 2×/day. **Starts day one and never stops.** |
| **Normalizer** | every 5 min | `swing normalize` | Drains `articles_raw`. Decoupled so an embedding-model change is a replay. |
| **Daily batch** | post-close | `swing batch --date` | prices → factors → swings → **sectors first**, then stocks → intraday capture → cluster → attribute → persist. |
| **Health check** | daily 09:00 | `swing health` | Per-source counts; alert on any source at 0 for 24h. |

Everything is reachable through one CLI so there is a single operational surface:

```
swing collect                      # run the daemon in the foreground
swing normalize [--limit N]
swing backfill prices [--years 2]
swing batch [--date YYYY-MM-DD] [--dry-run]
swing annotate [--blind]
swing eval placebo [--n 30|200]
swing query "why did NVDA move last Tuesday?"
swing health
```

---

# 6 — Build sequence

Ordered by the spec's gates. Each step lands as one commit with its tests.

### Step 0 — Day one, before any application code (≈2 hours)
This is the spec's irrecoverable mistake #1 and it costs one afternoon.

1. `git init`; install Python 3.12; `docker compose up -d` Postgres 16 + pgvector
2. `.env` from `.env.example`; Finnhub + Gemini keys; `SEC_USER_AGENT` with a real email
3. `articles_raw` table only — nothing else
4. `common/settings.py`, `common/timeutil.py`, `common/http.py`
5. `ingest/edgar.py` (8-K poller, CIK map from `company_tickers.json`, zero-padded to 10)
6. `ingest/news_rss.py` (12 IR feeds + WSJ/CNBC/MarketWatch)
7. `ingest/collector.py` + `ingest/health.py`; **start it and leave it running**

**Gate −1: rows accumulating in `articles_raw`, per-source counts visible.**
Everything below happens while this runs.

### Step 1 — Foundation (days 2–3)
Full `schema.sql`, `models.py`, `session.py`, alembic baseline · `store/queries.py` ·
`scripts/build_cik_map.py`, `scripts/discover_xbrl_tags.py` · all three config files ·
`tests/test_timeutil.py`. **Verify C1 here** — probe Finnhub's candle endpoint and confirm
whether yfinance 5m is the intraday path, before anything depends on it.

### Step 2 — Prices + normalization (days 3–5)
`ingest/prices.py` backfill (17 symbols × 2y) · `ingest/normalize.py` + first embeddings ·
`ingest/news_finnhub.py`, `analyst.py`, `news_marketaux.py`.
Verify SNDK starts **2025-02-24** and is not spliced with pre-2016 SanDisk.
**Gate 0: 2y of bars, zero missing trading days; per-source daily counts printing.**

### Step 3 — The analysis core (days 5–9) — *the highest-leverage work*
`factors.py` → `decompose.py` → `swings.py` → **`onset.py`** → `windows.py` ·
`ingest/intraday.py` · matplotlib residual-z plots for 5 tickers × 6 months ·
`tests/test_decompose.py`, `test_onset.py`, `test_windows.py`.
Check per-ticker R² and decide whether MU/SNDK need a second memory factor (data-sources A.2).
**Gate 1: every flagged spike looks like a real event; `onset_ts` verified by hand on 10 swings.**

### Step 4 — Retrieval (days 9–12)
`dedup.py` (both tiers) · `rank.py` · `novelty.py` heuristic · read the 20 largest clusters and
tune the MinHash/cosine thresholds **on real accumulated copy**, not synthetic tests.
**Gate 2: recall@10 ≥ 0.80 on the blind annotation subset.**
⚠️ Start the ≥30 blind annotations *now*, before tuning weights, so you are not anchored.

### Step 5 — The agent (days 12–15)
`schema.py` flat-first + `rehydrate()` · `llm.py` · `abstention.py` (+ exhaustive tests) ·
`validate.py` · `nodes.py` → `graph.py` · `prompts/attribution_v1.md`.
**Gate 3: 5/5 synthetic no-news cases return `unexplained`.**

### Step 6 — Evaluation (days 15–18)
`eval/annotate.py` (blind mode must not reveal clusters before commit) · `harness.py` ·
`placebo.py` + `cache.py`. Iterate on the 30-case smoke set; spend the full 200 once.
**Gate 4: confabulation < 10%, citation validity exactly 1.00.**

### Step 7 — ML (only after Gate 4)
Baselines first, every time: relevance-only, then LightGBM+TF-IDF, then the off-the-shelf encoder.
`models/splits.py` is shared so no training script can accidentally shuffle a time series.
**Gate 5: each component beats its baseline on a walk-forward split, or it does not ship.**

### Step 8 — Interface (days 22+)
`interface/batch.py`, `query.py`, `alert.py`, `cli.py`; unexplained-rate-by-ticker and
by-swing_type dashboards.

---

# 7 — Decisions still open

| # | Decision | Recommendation |
|---|---|---|
| D1 | Build in-place vs. nested `swing-agent/` subdir | **In-place** — the specs already live here |
| D2 | Postgres via Docker vs. local install | **Docker** (`pgvector/pgvector:pg16`) — nothing is installed locally, and this pins the pgvector version |
| D3 | Python 3.12 via `uv` / `pyenv` / `brew` | **`uv`** — fastest, and handles both the interpreter and the lockfile |
| D4 | Contact email for `SEC_USER_AGENT` | Needed before the EDGAR poller can run at all |
| D5 | Where the dead-feed alert goes (email / macOS notification / log-only) | **Log + macOS notification** to start; email needs SMTP config |
| D6 | Second memory factor for MU/SNDK | **Defer** to after Gate 1, as data-sources A.2 prescribes — decide on measured R² |

---

# 8 — Reconciling `swing-cli.md`

`swing-cli.md` specifies the command surface. It is compatible with this plan
with **one structural change**, which is much cheaper now (15 files) than later.

## 8.1 ⚑ The package-layout change — do this before Step 1

`swing-cli.md` Part 2 requires an installable package with a
`[project.scripts]` entry point. The current repo is a set of flat top-level
directories (`common/`, `ingest/`, `store/`, `analysis/`, `agent/`). Those work
only because `sys.path` happens to include the repo root. **Once pipx installs
this globally, `import common` and `import agent` become top-level names on the
user's system** — `common` in particular is highly collision-prone, and
`agent` is the exact failure `swing-cli.md`'s own troubleshooting section
anticipates (`ModuleNotFoundError: agent`).

Fix: one `src/`-layout package, everything under it.

```
stock-agent-analysis/
├── pyproject.toml                  # [project.scripts] swing = "swing.cli:main"
└── src/swing/
    ├── __init__.py
    ├── cli.py                      # dispatcher + REPL   (swing-cli.md Parts 5, 6)
    ├── commands.py                 # one function per subcommand
    ├── paths.py                    # SWING_HOME override  (swing-cli.md Part 4)
    ├── common/     settings, timeutil, http, logging, versioning
    ├── store/      schema.sql, models, session, queries, raw
    ├── ingest/     collector, edgar, news_rss, prices, normalize, health, ...
    ├── analysis/   factors, decompose, swings, onset, windows, dedup, rank
    ├── agent/      graph, state, nodes, schema, abstention, validate, llm
    ├── models/     train_*
    └── eval/       annotate, harness, placebo, cache
```

`swing-cli.md` puts `cli.py` inside `agent/`. Keep them separate: `agent/` is the
LangGraph attribution agent (§1 layer L4) and the CLI is layer L5. A CLI that
lives inside the agent package makes `swing coverage` — pure SQL, no LLM —
import the LangChain stack, which breaks the doc's own 300 ms startup target.

**Already satisfied:** Part 4's working-directory problem. `common/settings.py`
anchors on `REPO_ROOT` from `__file__` and pydantic-settings loads `.env` by
absolute path. Verified working from `~`. Only the `SWING_HOME` env override
still needs adding.

## 8.2 Command surface → module map

| Command | LLM | Backed by | Available after |
|---|---|---|---|
| `swing coverage` | No | `store/queries.py` | **now** (Step 0) |
| `swing collect` | No | `ingest/collector.py --once` | **now** (Step 0) |
| `swing health` | No | `ingest/health.py` | **now** (Step 0) |
| `swing why <T>` | No | `swings` + `attributions` | Step 5 |
| `swing stats <T>` | No | `daily_factors` + `swings` | Step 3 |
| `swing compare` | No | `daily_factors.r_squared` | Step 3 |
| `swing unexplained` | No | `attributions.verdict` | Step 5 |
| `swing ask` | **Yes** | insight agent | Step 5+ |
| `swing batch` | **Yes** | `interface/batch.py` | Step 5 |

Three commands are buildable today against `articles_raw`. The rest need tables
that do not exist yet — build the CLI skeleton early (it is the Step 4 gate in
`swing-cli.md`) but expect most subcommands to be stubs until Step 3–5.

## 8.3 ⚠️ Unresolved: `insight-agent-guide.md` does not exist

`swing-cli.md` cites it for the tool definitions, the guardrails, the
no-conversation-memory rationale, and the forecast refusal. It is not in the
repo, and these depend on it:

- `swing ask` — its whole tool surface
- `swing compare` — `idio_share`, a metric defined in neither existing spec
- `swing stats` — the `n`/small-sample warnings
- **The forecast guardrail** — `swing ask "is NVDA a buy?"` must refuse *before*
  calling Gemini. `agent-plan.md` says the system is not a predictor but
  specifies no refusal mechanism. This is new scope.

`idio_share` is presumably `|residual| / |total_return|` — the share of the move
that is idiosyncratic, computable from `daily_factors`. Confirm before building.

## 8.4 `swing collect` vs. the daemon

`swing-cli.md`'s cron form (`*/10 * * * * swing collect`) and the running daemon
are alternatives, not complements — running both double-polls every source.

Keep the daemon as primary; `swing collect` maps to `collector.py --once` for
manual/backfill use. **Note that cron does not solve the sleep problem** (§8.5):
a cron job does not fire while the machine is asleep either.

## 8.5 ⚠️ Operational: this Mac sleeps after 1 minute

`pmset` reports `sleep 1` and `powernap 0` on battery. `time.monotonic()` freezes
across macOS sleep, so the collector stops polling entirely — observed directly:
one poll, then 61 minutes with zero CPU accrued and no polls.

Current mitigation: the collector runs under `caffeinate -is`, which holds
`PreventUserIdleSystemSleep`. Verified cycling across a 9-minute idle window.

This is a workaround, not a fix. It keeps the Mac awake (battery cost) and does
not survive a lid close or reboot. For a component whose entire premise is
continuous collection, the real answer is an always-on host — a $5 VPS or a
Raspberry Pi — with Postgres and the collector on it. Everything else in this
system is a batch job that can run anywhere; only the collector must never stop.

---

# 9 — Step 1 findings (2026-08-31)

## 9.1 ✅ C1 resolved — intraday is capped at 60 days, hard

`yfinance` serves 5-minute bars for **the last 60 days only**. `period=2mo` and
`3mo` are rejected outright: *"The requested range must be within the last 60
days."* `1mo` returns 1,639 bars. Verified for NVDA, SMH and SPY — stock, sector
and market all available, all tz-aware (`America/New_York`), so `assert_utc`
converts cleanly.

Consequences, now fixed in `config/thresholds.yaml`:

- Onset detection is exact for swings **inside 60 days**.
- Older swings get `swing_type='unknown'`, `onset_source='fallback_48h'`, and a
  conservative 48-hour pre-move window. They are flagged and must be excluded
  from ranking-weight tuning.
- **Capture-on-detect is mandatory, not optional.** Persist a swing day's
  intraday bars the moment it is detected, or the window closes permanently.

`FINNHUB_API_KEY` is still empty, so the "is Finnhub's candle endpoint free?"
question is untested. It no longer blocks anything: yfinance is the plan of
record and it works.

## 9.2 ⚠️ The XBRL tag heuristic in the spec is wrong

`agent-plan.md` says to take the revenue tag with the most datapoints. Run
against the real watchlist, that picks:

- **`CostOfRevenue` for GOOGL** — a cost, not revenue
- **`CostOfRevenue` tied with `Revenues` for NVDA** — resolved by sort order
- **`SalesRevenueNet` for MU, AAPL and TTWO** — the pre-ASC606 tag retired
  around 2018. It wins on volume purely through history and has no recent data.

Five of twelve wrong. `scripts/discover_xbrl_tags.py` now excludes non-top-line
tags (`CostOf*`, `Deferred*`, `ContractWithCustomerLiability*`, ...) and ranks
by **most recent filing** rather than count. Every ticker now resolves to a tag
it actually files today, and the results are recorded per ticker in
`config/watchlist.yaml`:

| Tag | Tickers |
|---|---|
| `Revenues` | NVDA, QCOM, GOOGL, NFLX, TSLA, SBUX |
| `RevenueFromContractWithCustomerExcludingAssessedTax` | MRVL, MU, SNDK, AVGO, AAPL, TTWO |

An exact 6/6 split — direct confirmation that no single tag works.

## 9.3 ⚠️ Alembic autogenerate would have dropped the vector index

`alembic check` on the first pass proposed `remove_index` for all nine indexes
and `remove_constraint` for all ten CHECK constraints — including
`articles_embedding_idx`, the hnsw vector index. Cause: the ORM declared
columns but not indexes/constraints, so autogenerate read them as drift.

Anyone running `alembic revision --autogenerate` would have generated a
migration that silently destroys retrieval performance.

Fixed both ways: `include_object` in `env.py` excludes indexes from comparison
(schema.sql owns them, documented in-file), and the ten CHECK constraints are
now declared in `models.py` with the names Postgres assigns. `alembic check`
reports no drift, and a models-vs-`information_schema` comparison confirms all
13 tables match exactly.

## 9.4 Still open

- `FINNHUB_API_KEY` and `GEMINI_API_KEY` are empty. Needed for Step 2 (Finnhub
  news, analyst actions) and Step 5 respectively.
- `insight-agent-guide.md` still missing (see §8.3). `idio_share` has been
  implemented as `|residual| / |ret|` in `daily_factors`; confirm that matches
  the intent.

---

# 10 — API key verification (2026-08-31)

Both keys added and probed against the live APIs.

## 10.1 ⚠️ Finnhub free tier has NO price data at all

| Endpoint | Status | Note |
|---|---|---|
| `/quote` | 200 | o/h/l/c/pc, **no volume** |
| `/company-news` | 200 | 243 NVDA articles in one week — the Tier 2 workhorse |
| `/stock/recommendation` | 200 | analyst recommendation trend |
| `/calendar/earnings` | 200 | |
| **`/stock/candle`** | **403** | *both* intraday **and daily** |
| `/stock/price-target` | 403 | paid |

This is broader than C1 assumed. `agent-plan.md` §0.3 says "yfinance for the
historical backfill and Finnhub for ongoing updates" — **that ongoing-updates
path does not exist on the free tier.** yfinance is the source for *all* bars,
daily and intraday; Tiingo remains the fallback. `/quote` cannot substitute
because it carries no volume, and `volume_z` is required for §1.6 and for
turning an `unexplained` verdict into a flow-event signal.

Analyst actions (data-sources.md C.4) are therefore only half-covered:
recommendation trend yes, price targets no.

**Also:** Finnhub company-news mixes source tiers — the first sampled article
was SeekingAlpha, which is Tier 4 by our own rules. `normalize.py` must assign
tiers **per publisher, not per feed**, or aggregator content silently enters the
evidence pool.

## 10.2 ✅ The nested Attribution schema works — no flattening needed

`agent-plan.md` §3.1 and TROUBLESHOOTING both anticipate that
`Attribution -> candidates[] -> evidence[]` (three levels) may exceed Gemini's
supported JSON Schema subset, and prescribe flattening into two lists joined on
`candidate_id`.

Tested against both `gemini-3.7-flash` and `gemini-flash-latest`: **the nested
schema is accepted and returns correct output.** The flat variant and
`rehydrate()` are implemented and unit-tested, but kept as a dormant fallback so
the workaround does not have to be rediscovered later.

Behaviour on two hand-built cases was correct in both directions:

- **Real catalyst** (Tier 1 8-K, pre-move, guidance cut) → `explained`, `high`
  confidence, `event_type=guidance` (not `earnings` — it correctly identified
  guidance as the driver), cited only the pre-move cluster, and flagged the
  post-move CNBC story in `reactive_coverage_note` rather than as evidence.
- **Placebo** (irrelevant Starbucks/oil headlines against a real MRVL swing) →
  `unexplained`, zero candidates, with a sensible note.

Early Gate 3 signal, on a schema that was expected to need rework.

## 10.3 Model pinned, not floating

`.env` now sets `ATTRIBUTION_MODEL="gemini-3.7-flash"`.

`gemini-flash-latest` is a moving alias. Using it would make `attributions.model_id`
record an alias rather than the exact model, which defeats the entire point of
§3.1b: when a metric moves you could not tell whether it was your change or
Google rotating the model beneath you. Available flash-class models at time of
writing: `gemini-2.5-flash`, `gemini-3-flash-preview`, `gemini-3.1-flash-lite`,
`gemini-3.5-flash`, `gemini-3.6-flash`, `gemini-3.7-flash`, plus the `-latest`
aliases.

## 10.4 Integration detail worth not rediscovering

`langchain-google-genai` reads `GOOGLE_API_KEY` from the environment.
pydantic-settings loads our key into `Settings` and never exports it, so
construction fails with a misleading *"API key required for Gemini Developer
API"*. `agent/llm.py` passes it explicitly.

## 10.5 ⚠️ The dead-feed alert had a false negative — threshold was the bug

Docker Desktop stopped twice unprompted. The collector survived (per-job
exception handling worked as designed) but logged **441 failed jobs** across
roughly 20 hours of intermittent database outage, and `swing health` still
reported **"broken feeds: none"**.

The check was not wrong, its threshold was: a flat `stale_poll_hours=6` against
feeds that poll every 10–15 minutes. Six hours of silence is ~24 missed cycles,
and because RSS retains only the last 20–85 items, that is already permanent
loss before the alarm fires.

Fixed: the budget now derives from each feed's own interval,
`max(4 × poll_interval, 30 min)` — 40 min for a 10-minute feed, 1 h for a
15-minute feed, with a floor so a single transient blip does not alert. The
health job itself moved from every 6 h to every 30 min, since a 6-hourly check
would let a dead feed burn most of a 40-minute budget unobserved. Both are
covered by tests, including one asserting the health interval stays tighter
than the tightest budget.

This also means a database outage is now caught: a failed write means
`last_seen_at` stops advancing, which is exactly what the check reads.

## 10.6 ⚠️ Operational fragility — two independent single points of failure

Within one day: macOS slept the collector (§8.5), and Docker Desktop stopped
twice, taking Postgres with it. `StartDockerOnLogin` is not set, so Docker does
not even come back after a reboot.

Neither is a code problem and neither has a code fix. For the one component
whose premise is *continuous, unrecoverable-if-missed* collection, running it on
a laptop that sleeps, behind a Docker daemon that does not autostart, is the
weakest link in the system. **An always-on host is now the top operational
recommendation** — a $5 VPS or a Raspberry Pi running Postgres and the
collector. Everything else here is a batch job that can run anywhere.

Interim: enable *Start Docker Desktop when you sign in* in Docker settings, and
keep the `caffeinate -is` wrapper.

---

# 11 — Step 2 findings (2026-08-31)

## 11.1 ✅ Prices need no key; Gate 0 price condition met

yfinance is keyless. 17 symbols x 2 years = **8,468 unadjusted daily bars**,
505 per symbol, plus 98 dividends and 3 splits (NFLX 10:1 on 2025-11-17,
XLK and XLY 2:1 on 2025-12-05). **Zero missing trading days** across all 16
non-SPY symbols, checked against SPY as the calendar proxy.

Finnhub is not involved: `/stock/candle` is 403 on free for daily as well as
intraday, so yfinance is the source for all bars.

## 11.2 ⚠️ SNDK: when-issued trading, not a spliced predecessor

The history-start check fired. The cause was **not** the pre-2016 SanDisk the
spec warns about — it was **when-issued trading** in the 6 sessions from
2025-02-13, before regular-way trading opened on 2025-02-24. Median volume
406,800 against 9,678,900 after; prices in the post-spin range, whereas the old
SanDisk was acquired at ~$86.50 in 2016.

Same corrective action, different diagnosis. `enforce_history_start()` now
handles both and distinguishes them by gap size (>365 days = spliced
predecessor; otherwise when-issued). SNDK now starts exactly 2025-02-24 with
382 bars.

## 11.3 ⚠️ A regex bug was mistagging 41% of MU articles

Ticker aliases were derived from company names, which produced two defects:

1. `Take-Two Interactive` yielded the bare word **`Interactive`**, tagging any
   article about an interactive anything.
2. Worse, the alternation was **ungrouped**: `(?<!x)MU|Micron(?!x)` binds the
   lookbehind only to `MU` and the lookahead only to `Micron`. So `MU` matched
   with no trailing boundary at all — **"Musk", "Multiple" and "Munich" all
   tagged MU.** 28 of 68 MU articles were mistagged, on a ticker already flagged
   as thin on coverage.

Fixed: aliases are now curated in `config/watchlist.yaml`, never derived, and
the alternation is grouped. A test asserts every pattern is grouped, so a bare
`|` cannot reintroduce it.

## 11.4 ⚠️ Finnhub free news is 90% Tier 4, and has NO Tier 2 at all

583 company-news articles pulled across the watchlist. Resolved tiers:

| Publisher | n | Tier | |
|---|---|---|---|
| Benzinga | 317 | 4 | dropped |
| SeekingAlpha | 151 | 4 | dropped |
| CNBC | 59 | 3 | kept |
| ChartMill | 40 | 4 | dropped |
| Yahoo | 16 | 4 | dropped |

**Five distinct publishers. 524 of 583 dropped (90%). Zero Tier 2.**

No Reuters, AP, Bloomberg or Dow Jones. This contradicts `data-sources.md` C.3
("Tier 2 — Reuters/AP content arriving via Finnhub or Marketaux") and it leaves
a hole in the middle of the source-tier scheme: Tier 1 and Tier 3 are populated,
**Tier 2 is empty**.

Consequences worth deciding on:

- The only Finnhub content that survives is 59 CNBC articles **we already
  collect via RSS**. On the free tier, Finnhub company-news adds essentially
  nothing beyond the analyst recommendation trend.
- Design Rule 4 (dedup wire syndication) and the `distinct_sources`
  corroboration count were designed around wire copy that we do not have.
- `w_tier` in the ranking score spans a two-tier system, not four.

Options, cheapest first: add **Business Wire / PR Newswire / GlobeNewswire**
public RSS (free, carries the actual press releases, genuinely Tier 1-2); try
Marketaux for publisher breadth; or accept a two-tier system and retune
`w_tier`. Recommend the newswire RSS feeds — they fill the exact gap.

## 11.5 Packaging: extras must be installed into the tool env

`swing prices` failed from `~` with `No module named 'pandas'` while working
inside the project. The `uv tool` environment installs only base dependencies,
so the global command lacked the `[data]` extra. Fixed with
`uv tool install --editable ".[data,ml,llm]" --force`. Re-run that after adding
any dependency, or the global command silently diverges from the dev venv.

## 11.6 Fixed: Docker autostart

`AutoStart` was `false` in `settings-store.json`; now `true`, with a `.bak`
alongside. Docker Desktop may rewrite this file when it quits, so confirm in
Settings -> General -> "Start Docker Desktop when you sign in".

---

# 12 — Tier 2 is definitively unavailable on free tiers (2026-09-01)

All three free news sources now tested with live keys:

| Source | Tier 2 wire copy |
|---|---|
| Finnhub company-news | **0** — 5 publishers, none of them wire |
| Marketaux | **0** — reuters.com found=0, apnews.com 0, bloomberg.com 0, wsj.com 0 |
| Direct RSS | **0** — Reuters killed its public feeds |

`data-sources.md` C.3 assigns "Tier 2 — Reuters/AP content arriving via Finnhub
or Marketaux". **That content does not exist on these tiers.** C.1 already
concedes this for WSJ/FT/Dow Jones full text; it is in fact true of the entire
wire layer.

**This is a two-tier system.** Tier 1 (EDGAR 8-K, IR feeds, newswire press
releases) and Tier 3 (WSJ/CNBC/MarketWatch RSS) are well populated; Tier 2 is
structurally empty and will stay that way without a paid feed.

Consequences to handle in Step 4 (retrieval):

- `w_tier` in the §2.3 ranking score spans two populated tiers, not four. The
  `(5 - tier)` term gives Tier 1 = 4 and Tier 3 = 2; that is a reasonable
  2:1 weighting, so the formula still works — but tune it knowing the middle
  is empty rather than under-collected.
- `distinct_sources` as a corroboration count is weaker than designed. Wire
  syndication across dozens of outlets is the thing it was built to collapse,
  and there is none. Corroboration will mostly mean "the 8-K plus one or two
  tier-3 write-ups".
- Design Rule 4 (dedup before counting corroboration) still matters for tier-3
  rewrites of the same story, just far less than for wire copy.

This is not a gap to fix; it is the shape of a $0 stack. Worth revisiting only
if the project ever gets a budget for Benzinga-via-Polygon or RavenPack.

**Marketaux is therefore configured but not polled** (`enabled: false`,
`on_demand_only: true`). 1 of 15 sampled articles survived tiering and it
duplicated CNBC RSS.

---

# 13 — Price retrieval verified, and a Step 3 design change (2026-09-01)

## 13.1 ✅ Retrieval verified

- 8,462 daily bars, 17 symbols, read back through `store/queries.py`.
- **Zero** OHLC violations, non-positive prices, null closes, or bad volumes.
- **Cross-validated against Tiingo**: NVDA, MU, TSLA, SPY, SMH closes match the
  stored yfinance values to the cent (0.000% difference) on two dates.
- Latest NVDA close 220.78 independently matches the Finnhub live quote.
- Intraday: 78 five-minute bars per session, 09:30-15:55 ET, stored UTC.

## 13.2 Tiingo intraday has no volume; source choice is now date-dependent

Tiingo `/iex` 5-minute bars return exactly `date/open/high/low/close` — **no
volume field**. yfinance intraday has volume but only 60 days of history.

Neither gap blocks the plan: onset works on price returns alone (§1.3) and
`volume_z` comes from daily bars (§1.6). `fetch_intraday()` now takes yfinance
inside 60 days (free volume) and Tiingo beyond it (depth), so we get volume when
it is available and history when we need it.

## 13.3 ⚠️ The two-factor regression is collinear — orthogonalise the sector

`agent-plan.md` §1.1 regresses the stock on market and sector returns directly.
On real data that produces an **uninterpretable market/sector split**:

```
MRVL 2026-08-27, naive fit:
  beta_mkt = -1.20   beta_sec = +1.82
  market -0.78%      sector +5.57%      residual -6.87%
```

A negative market beta for a semiconductor is not credible. The cause is
collinearity — `corr(SPY, SMH) = 0.80`, `corr(SPY, XLK) = 0.89`. It is **not**
numerical instability: the betas are stable across refits (-1.18, -1.19, -1.16)
and R^2 is 0.57. The coefficients are jointly identified but individually
meaningless, which is the classic multicollinearity signature.

This matters because the split IS the user-facing output. §1.1's headline
sentence — *"Broad market accounts for -1.1%, semiconductor weakness for
-1.9%"* — is precisely the thing that becomes wrong.

**Fix: orthogonalise the sector factor against the market before fitting.**
Regress sector on market, keep the residual as the "pure sector" factor:

```
MRVL 2026-08-27, orthogonalised:
  beta_mkt = +3.42   beta_sec = +1.82
  market +2.24%      sector +2.48%      residual -6.87%
  corr(SPY, SMH_orth) = -0.000000
```

Three properties worth noting:

1. **The residual is bit-identical** (-6.8704% both ways). Swing *detection* is
   unaffected, so Gate 1 does not change.
2. The market/sector split becomes interpretable and correctly signed.
3. Components still sum to the total return.

`analysis/factors.py` must therefore fit in two stages, and store `g1` (the
sector-on-market loading) so the decomposition is reproducible.

## 13.4 ⚠️ `idio_share` can exceed 1

With offsetting factor components, `|residual| / |ret|` is 4.57 for the case
above (residual -6.87% against a -1.50% total move). Mathematically fine, but
"share" implies [0, 1] and a user reading "457%" will not trust it.

Options: clamp to [0, 1] for display; report it as a ratio and rename; or use a
variance-based definition (`var(residual) / var(ret)` over a window), which is
bounded and is the more standard "idiosyncratic share". **Recommend the
variance-based version** for `swing compare`, keeping the per-day ratio as a
separate raw field. Needs confirming against `insight-agent-guide.md`, still
missing.

---

# 14 — Step 3 complete: decomposition, swings, onset (2026-09-01)

## 14.1 What was built

| Module | Role |
|---|---|
| `analysis/factors.py` | Two-stage rolling fit -> `daily_factors`. Sectors first. |
| `analysis/decompose.py` | Single-day load + the user-facing sentence |
| `analysis/onset.py` | Locate the move onset from 5-min bars |
| `analysis/windows.py` | `swing_type` -> (pre_move, post_move) |
| `analysis/swings.py` | z-threshold, drift, volume_z, earnings_mode, capture-on-detect |
| `scripts/gate1_verify.py` | Residual plot + onset hand-check table |

CLI: `swing factors`, `swing detect`, `swing onsets`, and `swing stats` /
`swing compare` now backed by real data.

## 14.2 Results

**5,746 factor rows** across 15 entities (SNDK correctly gated to
`insufficient_history` at 382 bars < 400). **371 swings**: 361 daily, 10 drift.

Betas are credible after orthogonalisation — every `beta_mkt` positive, semis
high (MU 2.58, MRVL 2.51, NVDA 1.87), defensives low (NFLX 0.61, TTWO 0.69).
Swing rate 4.4-7.8% of days, right where |z| >= 2 should land under normality.

**This answers the open question in `data-sources.md` A.2**: MU's R^2 (0.643)
*beats* AVGO (0.560) and QCOM (0.477), so the memory-vs-logic concern does not
show up in the fit. **No second sector factor is needed.**

## 14.3 ✅ Gate 1

*"Every spike you flag must look like a real event; known events must appear."*
The top 10 swings by |z| all carry volume 4.6-28.3 sigma, and 5 of 10 are
flagged `earnings_mode` from an 8-K Item 2.02. Plot at `data/gate1_residuals.png`.

*"For 10 swings, verify by hand that `onset_ts` lands at the actual start of the
move and `swing_type` is correct."* Verified. `gap` (116) and `mixed` (151)
anchor at the session open, which is correct by definition. `intraday` (60)
spreads across 09:30-11:10.

Worked example — **TSLA 2026-05-11**, detected onset 10:25 ET:

```
09:30  CAR -1.35%
10:15  CAR -1.80%
10:30  CAR -2.51%   <- trough
11:15  CAR +0.54%
15:45  CAR +3.82%
```

The detected onset sits exactly at the trough before a +6.3% run. Onset
detection is correct on real data.

## 14.4 Bugs found and fixed during Step 3

- **`@dataclass(slots=True)` has no `__dict__`.** The swing UPSERT used
  `s.__dict__`, so detection ran, captured intraday, and silently wrote **zero
  rows**. Now `dataclasses.asdict`.
- **Capture-on-detect is wrong for a backfill.** 371 swings x 3 symbols is
  1,000+ sequential API calls; it timed out. Split into fast `detect_all()` and
  a resumable `backfill_onsets()`, with a `has_intraday()` check so the market
  and sector ETFs are not refetched once per ticker on the same date.
- **`idio_share` was unbounded.** Averaging `|residual|/|ret|` gave 1.18-3.29
  because days with near-zero `ret` dominate. `swing compare` now uses
  `var(residual)/var(ret)`, bounded [0, 1]. It tracks `1 - R^2` closely
  (MU 0.33 vs R^2 0.647), which cross-validates both. The raw per-day ratio is
  still stored on `daily_factors`.

## 14.5 Test coverage

76 tests. `test_decompose.py` builds series with **known** betas and asserts the
orthogonalised fit recovers them, that the orthogonalised factor is uncorrelated
with the market to 1e-9, and that **the residual is bit-identical** naive vs
orthogonalised. `test_onset.py` builds bar series whose onset is known by
construction, including one asserting that keying off the closing bar instead of
onset admits a post-move article — the bug itself, encoded as a test.

---

# 15 — Step 4: retrieval and dedup (2026-09-02)

## 15.1 What was built

| Module | Role |
|---|---|
| `analysis/dedup.py` | Single-link agglomerative clustering + MinHash merge |
| `analysis/rank.py` | The §2.3 score: semantic + timing + tier + novelty |
| `analysis/novelty.py` | Heuristic `1 - max_cos(trailing 30d)` |
| `analysis/retrieval.py` | swing -> windows -> clusters -> ranked -> persisted |
| `eval/annotate.py` | Blind + assisted annotation CLI |
| `eval/harness.py` | The five metrics |

CLI: `swing retrieve`, `swing annotate [--blind]`, `swing metrics`.

Built clusters for **472 swings**; **66 have pre-move coverage**.

## 15.2 Bugs found and fixed

- **pgvector types were never registered with psycopg**, so every `vector`
  column came back as a ~15KB string and clustering crashed on
  `float(...)`. `store/session.py` now calls `register_vector`, and
  `common/vectors.py` gives one coercion used by every consumer.
- **The novelty corpus contained the articles being scored.** `trailing_corpus`
  ended at `onset`, which is *after* the pre-move window starts, so every
  cluster matched itself at cosine 1.0 and novelty was uniformly **0.00** —
  silently disabling the signal while appearing to work. It now takes
  `exclude_after=window.start`; novelty is 0.05-0.31 on real data.
- **Ranking had no query vector**, so `semantic_relevance` returned a flat 0.5
  for everything and the score degenerated to timing + tier. On the NVDA
  2026-08-27 earnings swing the true catalyst (8-K Item 2.02) ranked **#5**,
  behind a GeForce NOW gaming press release that happened to be closer to the
  open. `retrieval.query_text()` now builds a move-anchored query
  ("<ticker> <name> stock fell X% — earnings, guidance, revenue, analyst
  rating..."); semantic scores spread 0.765-0.878 and the earnings release moved
  #6 -> #2, the 8-K #5 -> #3.
- **Tiingo had no retry or rate limiting.** A historical onset backfill fires
  hundreds of requests and the free tier 429s on bursts; `fetch_intraday_tiingo`
  returned 0 bars and swings silently fell back to the 48h window, which looks
  like missing data rather than throttling. Now paced at 1.2s with exponential
  backoff.

## 15.3 ⚠️ Gate 2 is blocked on article accumulation, not on code

Gate 2 wants recall@10 >= 0.80 over >= 30 **blind** annotations. Current
coverage makes that impossible to measure honestly:

| | |
|---|---|
| swings with any pre-move cluster | 66 of 472 |
| swings with **3+** pre-move clusters | **3** |
| swings since 2026-08-01 with coverage | 6 |
| cluster sizes | 140 singletons, 13 pairs, 3 triples, 1 quad |

Two consequences:

1. **recall@10 would be near-trivially 1.0.** With 1-2 clusters in a window,
   "is the true catalyst in the top 10" is answered by whether we hold the
   article at all — not by whether ranking works. The metric would look
   excellent and measure nothing.
2. **Dedup thresholds cannot be tuned yet.** The plan says to dump the largest
   clusters and read them, tuning on real syndication patterns. The largest
   cluster has 4 members. There is nothing to tune against, and — per §12 —
   there is no wire syndication in this stack to collapse anyway.

This is exactly what the plan predicted: *"your annotation set can only cover
dates where you hold articles"*, and *"build in strict phase order and you'll
arrive at evaluation in six weeks holding six weeks of news."* Live news capture
started 2026-08-30. EDGAR reaches back to 2023, which is why 66 swings have some
coverage, but news-side context does not exist for historical dates.

**The machinery is complete and tested; the data is three days old.**

## 15.4 What unblocks Gate 2

1. **Time.** Keep the collector running. Coverage grows every day; at ~200
   articles/day the 90-day Gate 0 target arrives around late November 2026.
2. **Blind annotations — yours to do.** `swing annotate --blind` shows the
   decomposition, swing_type, onset and volume_z, takes your independently
   researched answer, and only *then* reveals what retrieval returned. Do these
   before tuning ranking weights so you are not anchored. Budget ~10 minutes
   each; 30 of them is the difference between knowing recall and guessing it.
3. Then retune `w_semantic` / `w_timing` / `w_tier`. The NVDA case suggests
   **timing may be over-weighted for gap moves** — within an overnight window
   every article is equally "pre-move", so minutes-before-open is not evidence
   of causality, yet the 12h half-life gives a 09:00 press release 0.97 against
   0.37 for the prior evening's 8-K.

Step 5 (the attribution agent) does **not** depend on Gate 2 and can proceed:
its inputs are clusters, which now exist.
