# Stock Swing Attribution Agent — Build Instructions

**What you are building:** A system that detects unusual price moves in a stock, retrieves news from multiple journals, and produces an evidence-backed explanation of *why* it moved — or states that no catalyst can be identified.

**What you are not building:** A price predictor. The agent may answer a forward-looking question if asked, but forecasting is not what it is built, tuned, or evaluated for. Do not add prediction targets to the training loop.

---

# ⚠️ READ THIS FIRST — THE TWO IRRECOVERABLE MISTAKES

Almost everything in this plan can be fixed later. These two cannot, and both are easy to postpone because neither announces itself.

### 1. Start the news collector before you write anything else

Not the schema. Not the decomposition. The pollers, today.

Prices backfill in one call. **News does not.** RSS serves the last 20–50 items and free API tiers cap historical lookback, so every hour the collector isn't running is Tier 3 coverage that is gone permanently. This directly gates Phase 4 — your annotation set can only cover dates where you hold articles.

The EDGAR 8-K poller plus the twelve IR feeds is roughly 100 lines and covers a large share of your real catalysts on its own. Write that, start it, then come back and build the rest while it accumulates. See Phase −1.

### 2. Do not let onset detection slide to "later"

This is the only bug in this document that **passes every gate while producing wrong answers.** The timestamps all look correct, the tests pass, and the output is confidently wrong.

If you build Phase 2's windows off the closing bar because wiring up intraday bars is annoying, your system silently classifies post-move commentary as pre-move evidence. That is the reactive-journalism trap from Design Rule 2, rebuilt by accident, and it will not show up in any metric you have. See §1.3.

**Everything else here is recoverable if you get it wrong. One of these loses data permanently; the other corrupts your ground truth invisibly.**

---

# PART 1 — TECH STACK REQUIREMENTS

## 1.1 Framework decisions (read before installing anything)

### Use PyTorch. Do not use TensorFlow. Do not use both.

Pick PyTorch, for four reasons specific to this project:

1. Every model here is a text encoder or a small MLP. The pretrained encoders you will fine-tune live on HuggingFace, and HuggingFace is PyTorch-first — many checkpoints ship PyTorch weights only.
2. `sentence-transformers`, which you use for embeddings (Phase 2) and the bi-encoder fine-tune (Phase 5.3), is built directly on PyTorch. There is no equivalent TF path that isn't more work.
3. Nothing here needs TensorFlow's deployment tooling (TF Serving, TFLite). This runs as a batch job against a database.
4. Mixing both doubles your dependency surface and creates CUDA version conflicts for zero benefit.

If you already know TensorFlow and want it anyway, the only place it changes anything is §5.2, where you'd swap a two-line `transformers` call for a custom training loop. Not worth it.

### Use LangGraph, not the deprecated LangChain agent API.

LangChain and LangGraph hit v1.0 on 22 October 2025 and are complementary layers from the same company, not competitors:

- `create_agent` from LangChain is the fast path, and **it runs on the LangGraph runtime**.
- `StateGraph` from LangGraph is what you drop to for branching and custom control flow.

**For this project, go straight to `StateGraph`.** Your attribution flow has a real branch (abstain vs. explain) and a mandatory non-LLM validation node. The prebuilt loop does not fit.

**Critical:** `AgentExecutor` is deprecated and slated for removal. Most LangChain agent tutorials online still use it. If a tutorial doesn't mention LangGraph, close the tab — you're reading the legacy path.

### Add LightGBM. You will need it.

Two of the three ML components in Phase 5 must be benchmarked against a gradient-boosting baseline before you ship a neural net. On small tabular and small text-classification datasets, LightGBM frequently wins. This is not optional; it is a gate.

### Use Gemini Flash on the free tier. Do not build on Gemini Pro.

**Provider: Google AI Studio, free tier, no credit card.**

**Model: a Flash-class model. Not Pro.** Gemini Pro's free tier is trial-only — reported daily request caps have been as low as 50/day, which you would blow through in a single evaluation sweep. Flash and Flash-Lite carry far higher daily allowances and are the only sane target. Verify current limits at `ai.google.dev/gemini-api/docs/rate-limits` before you build; Google revised these quotas in December 2025 and third-party guides disagree on the numbers.

Three reasons Flash is the right call here, not a budget compromise:

1. **Your production volume is 1–3 requests per day.** At `|z| ≥ 2` across 12 tickers you'll trigger roughly one swing daily. That is orders of magnitude inside any free tier.
2. **Flash is arguably better for your core requirement.** Reasoning-tuned models show roughly a 24% *decrease* in abstention recall versus instruction-tuned counterparts — they hallucinate missing context and answer confidently instead of abstaining. Larger frontier models hallucinate when context is insufficient; smaller models abstain. Abstention is the whole ballgame for this system.
3. **Gemini uses constrained decoding** (`responseSchema`) to enforce schema at the token level, rather than best-effort instruction following.

**Run the attribution node with reasoning/thinking disabled or minimal, and `temperature=0`.** Longer reasoning chains improve accuracy but measurably degrade abstention recall. This is a direct trade-off and you want the abstention side of it. If you want more thinking in the system, put it in retrieval and ranking (Phase 2), not in the node that decides whether a story exists.

**No local LLM.** Earlier drafts suggested Ollama for bulk work; that's dropped. Gemini Flash's free daily allowance covers pre-labeling and classification work too, and running a 7B model locally isn't worth the RAM.

You do still run two small models locally, but neither is an LLM and both are lightweight:

| Local model | Size | Load |
|---|---|---|
| `bge-base-en-v1.5` (embeddings, Phase 2) | ~110M params, ~440 MB | Fine on CPU. Batch inference only. |
| DistilBERT-class classifier (Phase 5.2) | ~66M params | Fine-tunes on CPU in hours on a few thousand labels. |

If even those are too heavy, Gemini has an embedding endpoint you can swap in — but then embeddings become an API call per article and you'll want to watch your daily request budget. Try local first.

**Three free-tier gotchas:**

- **Prompts may be used for training.** That's the stated trade-off. Irrelevant here — you're sending public market data and public news headlines. Don't reuse the key for anything proprietary.
- **Limits are per Google Cloud project, not per key.** Extra keys don't help, and spinning up projects to dodge quotas violates Google's terms. Do not build key rotation into your ingest layer.
- **Keep the provider abstraction anyway.** If Flash's confabulation rate fails Gate 4, swapping should be one config line.

---

## 1.2 Runtime requirements

| Requirement | Value | Notes |
|---|---|---|
| Python | 3.12 | LangGraph 1.1.x dropped 3.9. Use 3.12 — 3.13/3.14 still have occasional wheel gaps in this dependency set. |
| PostgreSQL | 16+ | With the `pgvector` extension. |
| RAM | 16 GB | 8 GB works if you skip local embedding and use a hosted embedding API. |
| GPU | Not required for Phases 0–4 | Phase 5 fine-tuning runs on CPU in hours, or any consumer GPU in minutes. |
| Disk | ~10 GB | 2 years of bars + 90 days of articles for 40 tickers is small. Embeddings dominate. |

## 1.3 Dependencies

Create `requirements.txt` exactly as follows. These are minimum-version floors, not pins — see §1.5 step 3.

```
# --- Agent orchestration ---
langgraph>=1.0
langchain>=1.0
langchain-core>=1.0
langchain-google-genai>=2.0     # Gemini provider
google-genai>=1.0               # official Google SDK, pulled in as a dependency

# --- ML / embeddings (PyTorch only) ---
torch>=2.4
transformers>=4.44
sentence-transformers>=3.0
lightgbm>=4.3
scikit-learn>=1.5

# --- Data ---
numpy>=1.26,<3
pandas>=2.2
pyarrow>=16.0

# --- Storage ---
psycopg[binary]>=3.2
pgvector>=0.3
sqlalchemy>=2.0
alembic>=1.13

# --- Ingestion ---
finnhub-python>=2.4
yfinance>=0.2.40
feedparser>=6.0
httpx>=0.27
tenacity>=8.2                   # retry/backoff for rate-limited APIs

# --- Schema / validation ---
pydantic>=2.7
python-dotenv>=1.0
pyyaml>=6.0

# --- Dedup ---
datasketch>=1.6                 # MinHash/LSH for near-duplicate detection

# --- Dev ---
pytest>=8.0
matplotlib>=3.9                 # Phase 1 verification plots
```

### What each block is for

| Package | Role in this system |
|---|---|
| `langgraph` | The state machine: detect → decompose → retrieve → attribute → validate |
| `langchain-core` | Tool definitions (`@tool`), message types, structured output binding |
| `torch` | Backend for all embedding and fine-tuning work |
| `sentence-transformers` | Article embeddings (Phase 2), bi-encoder fine-tune (Phase 5.3) |
| `transformers` | Event classifier fine-tune (Phase 5.2) |
| `lightgbm` | Mandatory baseline for Phases 5.1 and 5.2 |
| `pgvector` | Vector similarity search inside Postgres — avoids running a second database |
| `datasketch` | MinHash LSH for the cheap deduplication pass |
| `tenacity` | Backoff against Finnhub's 60 calls/min limit |
| `pydantic` | The `Attribution` schema, which enforces honesty in Phase 3 |

## 1.4 API keys required

| Service | Purpose | Cost | Env var |
|---|---|---|---|
| Finnhub | Live quotes, company news | Free tier: 60 calls/min, websockets included | `FINNHUB_API_KEY` |
| Google AI Studio | The attribution step (Gemini Flash) | **Free tier, no credit card** | `GEMINI_API_KEY` |
| SEC EDGAR | 8-K filings, full-text search | Free, **no key** — but you must send a `User-Agent` header with a contact email or you get blocked | `SEC_USER_AGENT` |
| Tiingo *(optional)* | Clean end-of-day history for training | Free tier: 1,000 req/day | `TIINGO_API_KEY` |

`yfinance` needs no key and is fine for backfilling history. It's community-maintained and occasionally breaks — pin the version and keep Tiingo as a fallback.

## 1.5 Setup — run these in order

```bash
# 1. Environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 2. Install
pip install -r requirements.txt

# 3. Pin everything you just resolved
pip freeze > requirements.lock.txt

# 4. Database
createdb swing_agent
psql swing_agent -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 5. Verify PyTorch and the embedding model
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('BAAI/bge-base-en-v1.5'); \
print('embedding dim', m.get_sentence_embedding_dimension())"

# 6. Verify LangGraph is v1+
python -c "import langgraph; print('langgraph', langgraph.__version__)"

# 7. Verify Gemini connectivity and structured output
python -c "
from langchain.chat_models import init_chat_model
from pydantic import BaseModel
class Ping(BaseModel):
    ok: bool
    note: str
llm = init_chat_model('google_genai:gemini-flash-latest', temperature=0)
print(llm.with_structured_output(Ping).invoke('Reply ok=true, note=connected'))
"
```

**Step 7 must return a parsed `Ping` object**, not raw text. Replace the model string with whatever Flash-class model ID is current — check `ai.google.dev` rather than trusting a model name from any document, including this one. If step 7 fails with a quota error you've hit a per-project limit; if it fails with a schema error, see §3.1.

**Step 5 must print `embedding dim 768`.** That number must match the `vector(768)` column in §0.4. If you swap embedding models later you must migrate the column and re-embed everything — decide now.

**Step 6 must print 1.x.** If it prints 0.x, your library and your tutorials are out of sync.

## 1.6 Repository layout — create this now

```
swing-agent/
├── config/
│   ├── watchlist.yaml       # tickers → sector ETF mapping
│   ├── sources.yaml         # source → tier mapping
│   └── thresholds.yaml      # z-score cutoffs, time windows, ranking weights
├── ingest/
│   ├── prices.py            # Finnhub + yfinance collectors
│   ├── news.py              # Finnhub news + RSS
│   └── edgar.py             # SEC 8-K full-text
├── store/
│   ├── schema.sql
│   ├── models.py            # SQLAlchemy
│   └── queries.py
├── analysis/
│   ├── decompose.py         # ← the most important file in the repo
│   ├── swings.py
│   ├── dedup.py
│   └── novelty.py
├── agent/
│   ├── graph.py             # StateGraph definition
│   ├── tools.py
│   ├── schema.py            # Pydantic Attribution model
│   └── prompts.py
├── models/
│   ├── train_novelty.py
│   ├── train_events.py
│   └── train_retrieval.py
├── eval/
│   ├── harness.py
│   ├── placebo.py
│   └── annotate.py          # CLI for hand-labeling
├── requirements.txt
├── requirements.lock.txt
└── .env
```

---

# PART 2 — DESIGN RULES

These five rules drive the decisions below. Deviate only knowingly.

1. **Explain the residual, not the raw move.** Market and sector beta account for most of a stock's daily return. Strip them before attributing anything to news.
2. **Timing is evidence.** An article published before a move is a candidate catalyst. An article published after it is commentary — often a journalist reverse-engineering the same tape you're looking at.
3. **Abstention is a first-class output.** Stocks move on flows, rebalancing, and expiries with no news at all. An LLM handed a price move and a pile of articles will *always* produce a story unless you engineer against it.
4. **A syndicated wire story is one source, not thirty.** Deduplicate before counting corroboration.
5. **Every claim carries a citation with a timestamp.** No uncited narrative reaches the user.

---

# PART 3 — BUILD PHASES

Each phase ends with a gate. **Do not start the next phase until the gate passes.** Every phase after Phase 2 is only as trustworthy as the data beneath it.

| Phase | Deliverable | Gate |
|---|---|---|
| **−1** | **News collector running** | **Writing to `articles_raw` continuously — do this first** |
| 0 | Data layer | 2 years of bars, no gaps; sector entities configured |
| 1 | Decomposition + swing detection + onset | Swings match human judgment; onset verified on 10 cases |
| 2 | Retrieval + dedup | recall@10 ≥ 0.80 on the **blind** annotation set |
| 3 | Attribution agent | 5/5 synthetic no-news cases return `unexplained` |
| 4 | Evaluation harness | Confabulation rate < 10% on placebo test |
| 5 | ML components | Each model beats its stated baseline |
| 6 | Interface + alerting | End-to-end run on live data |

---

## PHASE −1 — START COLLECTING NEWS TODAY

**Do this before anything else in this document, including the database schema work you'd normally do first.**

Prices backfill trivially — two years arrive in one yfinance call. **News does not backfill.** RSS feeds serve only the last 20–50 items, and free news API tiers cap historical lookback. Every day you spend on decomposition without a collector running is a day of Tier 3 coverage you can never recover.

This gates Phase 4. Your 100–200 annotations can only cover dates where you have articles. Build in strict phase order and you'll arrive at evaluation in six weeks holding six weeks of news.

### Step −1.1 — Minimum viable collector

Write the smallest thing that persists articles. Skip embeddings, skip clustering, skip ticker tagging — those are backfillable from stored text. Only the raw capture is time-sensitive.

```sql
CREATE TABLE articles_raw (
  id           bigserial PRIMARY KEY,
  url          text UNIQUE NOT NULL,
  source       text NOT NULL,
  headline     text NOT NULL,
  summary      text,
  body         text,
  published_at timestamptz NOT NULL,
  retrieved_at timestamptz NOT NULL DEFAULT now(),
  raw          jsonb                      -- full API/feed payload, unparsed
);
```

**Store the full raw payload in `raw`.** You will change your mind about which fields matter, and reparsing stored JSON is free while refetching is impossible.

### Step −1.2 — Point it at these, in this order

1. **EDGAR 8-K poller** — 12 CIKs, every 10 minutes. Tier 1, highest value.
2. **Company IR RSS** — 12 URLs. Tier 1, trivial to add.
3. **Finnhub company news** — 12 tickers, daily.
4. **WSJ / CNBC / MarketWatch RSS** — every 15 minutes.

Steps 1–2 are about 100 lines and cover a large share of your real catalysts.

### Step −1.3 — Run it and leave it running

Cron or a systemd timer. Add one alert: if any source produces zero articles for 24 hours, email yourself. A silently dead feed for three weeks is three weeks of unrecoverable data.

Then go build Phase 0 while it accumulates.

> ### ✅ Gate −1
> Collector running continuously, writing to `articles_raw`, with per-source daily counts visible. **Start the clock before you do anything else.**

---

## PHASE 0 — Data layer

### Step 0.1 — Define the watchlist

Create `config/watchlist.yaml` with your 12 tickers and their sector ETF mapping. See the companion `data-sources-and-apis.md` for the full list and ticker-specific warnings.

```yaml
NVDA: {sector_etf: SMH, market_etf: SPY}
MU:   {sector_etf: SMH, market_etf: SPY}
AAPL: {sector_etf: XLK, market_etf: SPY}
TSLA: {sector_etf: XLY, market_etf: SPY}
```

### Step 0.1b — Treat sector ETFs as attribution entities too ⚠️

Six of your twelve tickers are semiconductors. When SMH itself drops 3%, all six residuals come back small and every ticker reports "no significant idiosyncratic move." Technically correct, completely useless — the actual story that day *is* the sector.

**Run the same pipeline on SMH, XLK, XLC, and XLY as first-class entities**, with SPY as their only factor:

```yaml
SMH: {sector_etf: null, market_etf: SPY, entity_type: sector}
XLK: {sector_etf: null, market_etf: SPY, entity_type: sector}
XLC: {sector_etf: null, market_etf: SPY, entity_type: sector}
XLY: {sector_etf: null, market_etf: SPY, entity_type: sector}
```

Retrieval for a sector entity searches industry-level news rather than a single company's ticker tag. Then a semiconductor swing can point at the sector attribution instead of generating six near-identical company explanations for one event.

This also fixes the ordering of your daily batch: **run sector entities first**, then individual names, so stock-level attributions can reference an already-computed sector story.

### Step 0.2 — Create the price schema

**Store unadjusted OHLCV plus a separate corporate actions table. Do not store adjusted close.** Adjusted series get silently rewritten every time a split or dividend occurs, so the series you pulled last month is not the series you pull today — and your backtests will change results for no visible reason. Apply adjustments at query time.

```sql
CREATE TABLE bars (
  ticker  text NOT NULL,
  ts      timestamptz NOT NULL,
  open    numeric, high numeric, low numeric, close numeric,
  volume  bigint,
  PRIMARY KEY (ticker, ts)
);

CREATE TABLE corporate_actions (
  ticker  text NOT NULL,
  ex_date date NOT NULL,
  kind    text NOT NULL,   -- 'split' | 'dividend'
  ratio   numeric,         -- 2.0 for a 2:1 split
  amount  numeric          -- cash dividend per share
);
```

### Step 0.3 — Ingest prices

Pull 2 years of daily OHLCV for every watchlist ticker **plus every sector ETF plus SPY**. You need a full year before your analysis window just to fit rolling betas.

Use `yfinance` for the historical backfill and Finnhub for ongoing updates. Wrap all Finnhub calls in `tenacity` retry with exponential backoff — you will hit the 60/min limit.

### Step 0.4 — Create the article schema

```sql
CREATE TABLE articles (
  id           bigserial PRIMARY KEY,
  url          text UNIQUE NOT NULL,
  source       text NOT NULL,          -- 'reuters', 'wsj', 'sec-edgar'
  source_tier  int  NOT NULL,          -- 1..4
  headline     text NOT NULL,
  body         text,
  published_at timestamptz NOT NULL,   -- source-reported, UTC
  retrieved_at timestamptz NOT NULL,   -- when YOU saw it
  tickers      text[] NOT NULL,
  embedding    vector(768),
  minhash      bytea,
  cluster_id   bigint
);

CREATE INDEX ON articles USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON articles (published_at);
CREATE INDEX ON articles USING gin (tickers);
```

**Store both `published_at` and `retrieved_at`.** They diverge, and the gap is meaningful: an article retrieved 6 hours after its stated publication time is weaker timing evidence than one seen within a minute.

**Normalize every timestamp to UTC on write.** Add an assertion that rejects naive datetimes. Mixed timezones are the single most common silent bug in this kind of pipeline, and the failure mode — pre-move articles misclassified as post-move — corrupts your core signal invisibly.

### Step 0.5 — Set source tiers

Create `config/sources.yaml`. This drives retrieval weighting and citation display everywhere downstream.

| Tier | Contents | Treatment |
|---|---|---|
| 1 | SEC filings (8-K, 10-Q, 10-K), company press releases, earnings transcripts | Primary evidence. Unambiguous timestamps. |
| 2 | Reuters, AP, Bloomberg, Dow Jones newswire | Strong, but heavily syndicated — dedup matters most here. |
| 3 | WSJ, FT, Barron's, CNBC, sector trade press | Analysis and framing. Often the most *interesting* content. |
| 4 | Aggregators, SEO content farms, unattributed blogs | **Exclude. Do not ingest.** |

**Prioritize the EDGAR 8-K feed.** It is the legally required disclosure channel for material events, it is free, its timestamps are reliable, and it is very often the actual catalyst that everything in Tier 2 and 3 is merely reporting on. Send a `User-Agent` header with a contact email or SEC will block you.

### Step 0.6 — Embed on ingest — headline and summary only ⚠️

Run every article through `bge-base-en-v1.5` at write time and store the vector. Batch in groups of 32.

**Embed `headline + summary` only. Never embed the body.**

This is not a performance optimization, it's a correctness fix. Your Tier 2 sources give you full text; WSJ — your marquee Tier 3 source — gives you a headline and one line. If relevance scoring runs over body text, full-text sources will systematically outrank WSJ simply because they have more tokens to match on. Your carefully designed source tiers get silently inverted by an artifact of what each publisher lets you scrape.

Uniform headline-plus-summary embedding puts every source on equal footing. Pass body text to the LLM as supplementary context when you have it, but **never let it influence ranking**.

Corollary: make `articles.body` nullable and confirm retrieval, clustering, and the agent prompt all work on headline-only records. Architect around full text and every premium source blocks you.

> ### ✅ Gate 0
> - 2 years of bars for all tickers and ETFs, zero missing trading days
> - 90 days of articles
> - A monitoring script that prints per-source article counts per day — this is how you catch a feed that has silently died

---

## PHASE 1 — Return decomposition and swing detection

This is the highest-leverage phase in the project. **Get it right before touching the LLM layer.**

### Step 1.1 — Implement the decomposition

Fit a rolling regression of the stock's daily return on market and sector returns:

```
r_stock = α + β_mkt · r_market + β_sector · r_sector + ε
```

Fit on a trailing 120-day window ending the day **before** the event. **Never include the event day in its own beta estimate.**

```python
import numpy as np

def decompose(stock_rets, mkt_rets, sector_rets, lookback=120):
    """Returns (residual, market_component, sector_component) for the last day."""
    y = stock_rets[-lookback-1:-1]
    X = np.column_stack([
        np.ones(lookback),
        mkt_rets[-lookback-1:-1],
        sector_rets[-lookback-1:-1],
    ])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, b_mkt, b_sec = coef
    mkt_c = b_mkt * mkt_rets[-1]
    sec_c = b_sec * sector_rets[-1]
    residual = stock_rets[-1] - alpha - mkt_c - sec_c
    return residual, mkt_c, sec_c
```

This output feeds the user-facing explanation directly: *"NVDA fell 3.5%. Broad market accounts for −1.1%, semiconductor weakness for −1.9%. The stock-specific residual is −0.5%."*

That sentence is what prevents the system's most common failure — blaming a company headline for a sector-wide selloff.

### Step 1.2 — Detect swings on the residual

Threshold the **residual**, normalized by its own trailing volatility:

```python
z = residual / np.std(residual_history[-60:])
is_swing = abs(z) >= 2.0
```

**Do not use a fixed percentage threshold.** A 4% move is an earthquake for a utility and an ordinary day for a small-cap biotech. The z-score adapts automatically.

### Step 1.3 — Locate the move onset ⚠️ critical

**Do not skip this.** Detecting swings on daily bars while classifying articles by hour-level timestamps is a silent correctness bug, and it will pass every gate in this document while producing wrong answers.

If NVDA closes down 3.5%, *when* did it fall? If the drop happened at 9:35am and the stock drifted sideways afterward, a 2pm article is **post-move commentary** — but naive code sees "published before the closing bar timestamp" and files it as a candidate catalyst. That is precisely the reactive-journalism trap from Design Rule 2, rebuilt by accident.

Pull 5-minute bars for swing days (Finnhub's free tier covers intraday) and find where the cumulative abnormal return actually starts running:

```python
def find_onset(intraday_bars, prev_close, sector_intraday, beta_sector):
    """Return (onset_ts, swing_type). Onset = start of the largest sustained CAR run."""
    r = np.diff(np.log(intraday_bars.close), prepend=np.log(prev_close))
    ar = r - beta_sector * sector_intraday          # abnormal return per bar
    car = np.cumsum(ar)

    gap_ar = ar[0]                                   # overnight, close → open
    intraday_move = car[-1] - car[0]

    if abs(gap_ar) > 0.7 * abs(car[-1]):
        return intraday_bars.ts[0], "gap"
    if abs(gap_ar) > 0.3 * abs(car[-1]):
        return intraday_bars.ts[0], "mixed"

    direction = np.sign(intraday_move)
    onset_idx = int(np.argmin(direction * car))      # trough before the run
    return intraday_bars.ts[onset_idx], "intraday"
```

**Set `T = onset_ts`, not the closing timestamp.** Every window in Phase 2 keys off this.

### Step 1.4 — Classify the swing type

The three types need different retrieval windows, because the catalyst lives in a different place for each:

| `swing_type` | Meaning | Retrieval window |
|---|---|---|
| `gap` | Move happened close-to-open | Previous close → today's open. **Nothing intraday.** |
| `intraday` | Move happened during the session | `onset − 24h` → `onset` |
| `mixed` | Both | Previous close → `onset` |

**Gap moves are the easiest to attribute correctly** — the window is narrow and usually contains an 8-K or a press release. Getting this classification right is cheap accuracy, so build it before tuning anything else.

Store `swing_type` and `onset_ts` on the swing record. Both go into the agent prompt.

### Step 1.5 — Add a multi-day drift detector

Single-day thresholding misses slow-developing stories. A stock bleeding 1.3σ a day for four days never fires, and that's often the more interesting narrative.

Run a second detector on cumulative abnormal return over rolling windows:

```python
for window in (3, 5):
    car = residual_history[-window:].sum()
    car_z = car / (np.std(residual_history[-60:]) * np.sqrt(window))
    if abs(car_z) >= 2.5:            # higher bar than single-day
        emit_swing(kind="drift", window=window, z=car_z)
```

Drift swings get a retrieval window spanning the whole run, and the prompt should ask for a *developing* narrative rather than a single catalyst. **Deduplicate against single-day swings** — one big day inside a 5-day window is the same event, not two.

### Step 1.6 — Capture volume context

You already ingest volume. It costs nothing and it rescues your dead-end verdicts:

```python
volume_z = (volume - volume_ma20) / volume_std20
```

A large residual on 3× normal volume with no news is a **flow event** — index rebalancing, a block trade, a position unwind. The same move on normal volume is closer to noise. Right now both collapse to `unexplained`, which tells the user nothing.

Store `volume_z` on the swing record and pass it to the agent. §3.2 uses it to make the unexplained note actionable.

### Step 1.7 — Handle earnings properly, don't just suppress them

An earlier draft said to exclude earnings dates. **That was wrong.** Earnings are the single largest driver of idiosyncratic moves — suppressing them means going dark on the four days a year per ticker that matter most.

The honest constraint: proper earnings attribution needs *surprise versus consensus*, and consensus estimates are not free (Zacks, Refinitiv, and Finnhub's estimate endpoints are all paid). So don't pretend to compute surprise.

**Build an `earnings_mode` branch instead.** On a swing coinciding with an 8-K Item 2.02, the agent's job changes from "find the catalyst" — you already know it — to "characterize it." From free sources you can compute:

- **Sequential and year-over-year deltas** from XBRL `companyfacts` (revenue, margin, EPS, segment detail)
- **Guidance language diff** — pull the Item 2.02 exhibit and diff the forward-looking section against the prior quarter's

Guidance changes move stocks more than the reported quarter does, and the diff is free. Output reads: *"Revenue +12% YoY. Guidance revised from $X to $Y, down from prior quarter's range."* That's honest and useful, versus a suppressed date that explains nothing.

Still exclude **index rebalance dates and quad-witching** — structural, unrelated to journalism, nothing to characterize.

> ### ✅ Gate 1
> Plot 6 months of residual z-scores for 5 tickers with `matplotlib`. Every spike you flag must look like a real event to a human eye, and known events (a big earnings miss, an FDA decision) must appear.
>
> **Additionally:** for 10 swings, verify by hand that `onset_ts` lands at the actual start of the move and `swing_type` is correct. If onset detection is wrong, everything in Phase 2 is wrong.

---

## PHASE 2 — Retrieval and deduplication

### Step 2.1 — Define the candidate windows

**`T` is `onset_ts` from §1.3, not the closing timestamp.** If you skipped onset detection, go back — everything below is wrong without it.

Windows depend on `swing_type`:

| `swing_type` | Pre-move (candidate catalysts) | Post-move (reactive coverage) |
|---|---|---|
| `gap` | Previous session close → today's open | Today's open → open + 24h |
| `intraday` | `onset − 24h` → `onset` | `onset` → `onset + 24h` |
| `mixed` | Previous close → `onset` | `onset` → `onset + 24h` |
| `drift` | Window start − 24h → window end | Window end → +24h |

**Never merge the pre and post buckets.** This distinction is the backbone of the system's honesty. Financial journalism is largely written after the move — "stock slides on concerns about Y" is frequently a reporter reverse-engineering a story from the tape, doing exactly what your agent does, with the same guesswork. Retrieve without this split and you build a circular attribution machine.

Note the gap window is *tighter* than the old blanket 72 hours, and that's the point: for a gap move you know the catalyst arrived overnight, so a narrow window raises precision at no cost to recall.

### Step 2.2 — Deduplicate in two passes

Reuters and AP copy is syndicated near-verbatim across dozens of outlets. Without dedup, your agent reports "eight sources corroborate this" when it is one story reprinted eight times.

1. **Cheap pass:** MinHash LSH via `datasketch` on the article body. Jaccard ≥ 0.9 means near-identical. Catches straight reprints.
2. **Semantic pass:** agglomerative clustering on embeddings, cosine ≥ 0.90. Catches rewrites of the same underlying story.

Then compute per cluster:

```python
cluster = {
    "id": ...,
    "canonical_article": ...,   # earliest published, best tier
    "member_count": ...,
    "distinct_sources": ...,    # ← this is your corroboration count
    "earliest_published": ...,  # ← this is your timing evidence
    "tier": ...,                # best (lowest) tier in the cluster
}
```

**The agent reasons over clusters, never over raw articles.** Pass clusters to the LLM. Never pass the raw article list.

### Step 2.3 — Rank clusters

```
score = w_sem  · semantic_relevance
      + w_time · timing_score        # decays with distance before T; 0 if after T
      + w_tier · (5 − tier)
      + w_nov  · novelty             # Phase 5; set to 0.0 until then
```

Start with `w_sem=1.0, w_time=1.0, w_tier=0.5, w_nov=0.0` in `config/thresholds.yaml`. Tune on annotated cases in Phase 4.

> ### ✅ Gate 2
> Manually pick 20 swings where you know the true cause. The correct cluster must appear in the top 10 at least **80%** of the time. If it does not, no amount of prompt engineering downstream will save you — fix retrieval first.

---

## PHASE 3 — The attribution agent

### Step 3.1 — Write the schema before the prompt

The schema is what enforces honesty. The prompt only encourages it. Put this in `agent/schema.py`:

```python
from pydantic import BaseModel, Field
from typing import Literal

class Evidence(BaseModel):
    cluster_id: int
    headline: str
    source: str
    source_tier: int
    published_at: str
    timing: Literal["pre_move", "post_move"]
    distinct_sources: int
    relevance_note: str = Field(description="Why this connects to the move")

class Candidate(BaseModel):
    catalyst: str = Field(description="One sentence: what happened")
    event_type: Literal[
        "earnings", "guidance", "analyst_action", "m_and_a",
        "regulatory", "litigation", "product", "macro",
        "management", "other",
    ]
    direction_consistent: bool = Field(
        description="Would this plausibly move the stock in the observed direction?"
    )
    magnitude_plausible: bool = Field(
        description="Is a move of this size plausible for this event type?"
    )
    evidence: list[Evidence]
    confidence: Literal["high", "medium", "low"]

class Attribution(BaseModel):
    ticker: str
    date: str
    total_return: float
    market_component: float
    sector_component: float
    residual: float
    residual_z: float

    # --- swing context (from Phase 1) ---
    swing_type: Literal["gap", "intraday", "mixed", "drift"]
    onset_ts: str
    volume_z: float
    earnings_mode: bool

    verdict: Literal["explained", "partially_explained", "unexplained"]
    candidates: list[Candidate]
    unexplained_note: str | None
    source_disagreement: str | None = Field(
        description="Where outlets frame the same event differently"
    )
    reactive_coverage_note: str | None = Field(
        description="Post-move commentary, explicitly flagged as after-the-fact"
    )
```

### Step 3.1b — Version every stored attribution ⚠️

Add these to the **database table**, not the LLM output schema — you populate them, not the model:

```sql
ALTER TABLE attributions
  ADD COLUMN prompt_version text NOT NULL,      -- 'v3', bump on every prompt edit
  ADD COLUMN model_id       text NOT NULL,      -- exact model string used
  ADD COLUMN config_hash    text NOT NULL,      -- hash of thresholds.yaml + ranking weights
  ADD COLUMN created_at     timestamptz NOT NULL DEFAULT now();
```

Without these you have an unfalsifiable system. When you edit the prompt or Google rotates the Flash model, every prior record came from a different machine — and your Phase 5.3 training pairs become a blend of system versions you can't separate. You won't be able to tell whether a metric moved because of your change or theirs.

Compute `config_hash` from the actual config files at runtime, not by hand. Cheap now, impossible to reconstruct later.

Confidence semantics — apply consistently:

- **high** — Tier 1 primary source, pre-move, direction *and* magnitude both plausible
- **medium** — Tier 2–3 pre-move coverage, multiple distinct sources
- **low** — single source, or ambiguous timing, or implausible magnitude

### ⚠️ Gemini schema depth — test this before you build on it

Gemini's supported JSON Schema subset is narrower than OpenAI's, and nesting beyond three levels raises the error rate. `Attribution → candidates[] → evidence[]` sits right at that boundary.

**Test the real schema first**, right after setup step 7:

```python
llm.with_structured_output(Attribution).invoke("<a realistic prompt>")
```

If it fails or returns malformed output, flatten rather than fight it. Return two flat lists and join them in Python:

```python
class AttributionFlat(BaseModel):
    # ... scalar fields unchanged ...
    candidates: list[CandidateFlat]     # each has a candidate_id, no nested evidence
    evidence: list[EvidenceFlat]        # each carries candidate_id as a foreign key
```

Reassemble into the nested `Attribution` after parsing. You keep the same validation guarantees, and the join is five lines. Do this before writing prompts — discovering it in Phase 4 means redoing your eval runs.

### Step 3.2 — Enforce abstention in code, not in the prompt

**This is the most important function in the agent.** Prompts are suggestions; code is a guarantee. Run this *after* the LLM returns.

```python
def enforce_abstention(attr: Attribution) -> Attribution:
    valid = [
        c for c in attr.candidates
        if any(e.timing == "pre_move" and e.source_tier <= 3 for e in c.evidence)
        and c.direction_consistent
    ]
    if not valid:
        attr.verdict = "unexplained"
        attr.candidates = []
        attr.unexplained_note = _unexplained_note(attr.volume_z)
    elif attr.residual_z > 3.0 and all(c.confidence == "low" for c in valid):
        attr.verdict = "partially_explained"
    return attr


def _unexplained_note(volume_z: float) -> str:
    """Volume turns a dead-end verdict into a usable signal."""
    base = "No pre-move catalyst identified from credible sources. "
    if volume_z >= 2.0:
        return base + (
            f"Volume was {volume_z:.1f}σ above normal, consistent with a flow "
            "event — index activity, a block trade, or a position unwind — "
            "rather than news."
        )
    if volume_z <= -1.0:
        return base + (
            "Volume was below normal, so the move may reflect thin liquidity "
            "rather than a catalyst."
        )
    return base + (
        "Volume was unremarkable. The move may reflect positioning or "
        "information not captured in the monitored sources."
    )
```

Without the volume branch, a flow event and a genuine coverage gap produce identical output. With it, the first is a real answer and the second is a prompt to add sources.

### Step 3.3 — Build the StateGraph

```
detect_swing
    │
decompose_return
    │
    ├──► residual_z < 2.0 ──► emit "no significant idiosyncratic move" ──► END
    │
retrieve_clusters
    │
    ├──► zero pre-move clusters ──► emit "unexplained" ──► END
    │
llm_attribute          (structured output bound to Attribution)
    │
enforce_abstention     (pure Python, no LLM)
    │
validate_citations     (every cited cluster_id must exist in the retrieved set)
    │
END
```

Two LangGraph specifics that will bite you:

- Annotate any message-list field in your state with the `add_messages` reducer. Forgetting it is the most common LangGraph bug — the agent loses context between nodes.
- Use `TypedDict` for the state schema. Pydantic works, but `TypedDict` has lower overhead and plays cleanly with checkpointing.

**`validate_citations` is not optional.** Check deterministically that every `cluster_id` the model cites actually appeared in what you passed it. Models occasionally invent plausible-looking IDs.

### Step 3.4 — Configure the model

Put this in `agent/graph.py` and never override it per-call:

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    settings.ATTRIBUTION_MODEL,   # "google_genai:<current-flash-model>"
    temperature=0,                # reproducible eval runs
)
attributor = llm.with_structured_output(Attribution)
```

**Disable thinking/reasoning on this node** if the model exposes a budget parameter. Higher reasoning budgets improve accuracy but degrade abstention recall, and abstention is what this node exists to get right.

**`temperature=0` is not optional.** Your Phase 4 harness reruns the same 200 placebo cases after every prompt change. With nonzero temperature you cannot tell whether a metric moved because of your edit or because of sampling noise.

Wrap the call in `tenacity` retry on 429. Free-tier quotas throttle by requests-per-minute as well as per-day, and a batch job that fires several attributions back to back will trip the RPM limit.

### Step 3.5 — Write the prompt

- Pass clusters as a numbered list with explicit `published_at`, `source`, `tier`, and `timing`. Make timing impossible to overlook.
- State the residual decomposition in the prompt. Give the model the number it is actually explaining — not the headline move.
- Include verbatim: *"If the pre-move articles do not plausibly account for a move of this magnitude and direction, return an empty candidate list. Producing no explanation is a correct and expected outcome."*
- Require `source_disagreement`. When FT frames a move as a margin problem and WSJ frames it as demand, that divergence is the most valuable thing the system can surface. Do not let it get averaged into mush.

> ### ✅ Gate 3
> Run on 10 real swings and 5 synthetic no-news cases (large residual, article pool scrubbed of anything relevant). **All 5 synthetic cases must return `unexplained`.**

---

## PHASE 4 — Evaluation

**Build this before Phase 5.** Without it you cannot tell whether any model you train helps or hurts.

### Step 4.1 — Build the annotation CLI, and annotate blind ⚠️

Hand-label 100–200 swings with the true catalyst. Tedious, unavoidable, and it is the asset the whole project rests on.

**The trap:** if you label by picking the correct cluster *from the retrieved list*, you can never record a catalyst that retrieval missed. Recall@10 then looks excellent by construction and Gate 2 measures nothing. This is circular and it will fool you.

**Two annotation modes, and you need both:**

| Mode | Process | Measures |
|---|---|---|
| **Assisted** (~170 cases) | Show retrieved clusters, pick the right one | Attribution accuracy. Fast. |
| **Blind** (≥30 cases) | Determine the cause *independently* — read that day's coverage yourself, check the 8-K, search the web — **then** reveal what retrieval returned | **Recall. The only honest measurement you have.** |

Do the blind set first, before you've tuned ranking weights, so you aren't anchored. Blind cases are slow — maybe 10 minutes each — but 30 of them is the difference between knowing your recall and guessing it.

Record blind cases with a `blind: true` flag and compute recall@10 **only** over those.

`eval/annotate.py` should show the residual chart, `swing_type`, `onset_ts`, and `volume_z`, and in blind mode must not display retrieved clusters until you've committed an answer.

### Step 4.2 — Implement the metrics

| Metric | Definition | Target |
|---|---|---|
| Retrieval recall@10 | True catalyst cluster in top 10 | > 0.85 |
| Attribution accuracy | Top candidate matches annotation | > 0.70 |
| Abstention precision | Of `unexplained` verdicts, share that truly had no catalyst | > 0.80 |
| **Confabulation rate** | Of no-catalyst cases, share that got a confident explanation | **< 0.10** |
| Citation validity | Cited clusters existing in the retrieved set | 1.00 |

**Confabulation rate is the metric that matters most.** A system that explains 90% of moves correctly and invents stories for the other 10% is worse than one that explains 70% and abstains — because you cannot tell which bucket any given answer is in.

### Step 4.3 — Build the placebo test

Take a real price move. Feed the agent articles from a **randomly chosen different week** for the same ticker.

- Correct behavior: `unexplained`
- Incorrect behavior: a beautiful, fully-cited, entirely fabricated narrative

```python
def placebo_case(ticker, real_date, rng):
    swing = get_swing(ticker, real_date)
    fake_window = rng.choice(other_weeks(ticker, exclude=real_date))
    clusters = get_clusters(ticker, fake_window)   # wrong articles, real swing
    return run_agent(swing, clusters)              # must return "unexplained"
```

Run 200 cases. Above 10% confabulation means the abstention path is broken and nothing downstream is trustworthy.

**Budget this against your free-tier quota.** 200 cases is 200 requests, and you will rerun the sweep after every prompt change. On a Flash-class model that fits in a day; on Pro it would take a week. Two things make this manageable:

- **Cache the retrieval side.** Build each placebo case's cluster payload once and persist it. Re-running after a prompt tweak should hit the LLM only, never re-fetch or re-embed.
- **Use a 30-case smoke set for iteration** and save the full 200 for when you think you're done. Most prompt bugs show up in the first ten cases.

Your evaluation, not your production traffic, is what consumes the quota. Production is one to three swings a day; a single eval sweep is 200 requests. Plan around the sweep.

### Step 4.4 — Account for LLM contamination

Your LLM was trained on data covering your historical test period. When you ask it to explain a well-known 2024 move, part of the answer may be recall rather than inference from the articles you supplied.

Two mitigations: weight evaluation toward obscure tickers and minor moves unlikely to be memorized, and treat the placebo test — which cannot be passed by recall — as your primary honesty signal.

> ### ✅ Gate 4
> Confabulation rate < 10% across 200 placebo cases. Citation validity exactly 1.00.

---

## PHASE 5 — ML components

Three models, in this order. **Each must beat its stated baseline on held-out data or it does not ship.**

### Step 5.1 — Novelty scoring (build first)

Prices react to *surprise*, not information. An article restating three weeks of existing coverage is not a catalyst even when it is highly relevant.

Start with a heuristic — no training required:

```python
novelty = 1.0 - max_cosine_similarity(cluster_embedding, trailing_30d_corpus)
```

Then, once you have annotations, train a small PyTorch MLP on
`[novelty, tier, timing_delta, distinct_sources, semantic_relevance]`
to predict "is this the true catalyst." Clean supervised problem, real labels, no forecasting.

**Baseline to beat:** raw semantic relevance alone. Measure recall@10 before and after adding novelty to the §2.3 ranking score. If it doesn't move, keep the heuristic and skip the MLP.

### Step 5.2 — Event classification

Fine-tune a small encoder with `transformers` (DistilBERT-scale is plenty) to classify clusters into the `event_type` taxonomy from §3.1.

Payoffs: structured filtering, and empirical base rates. With a few thousand labeled events you can compute *"guidance cuts in this sector have historically moved stocks 4–7%"* — which turns `magnitude_plausible` from an LLM guess into a data-backed check.

**Baseline to beat:** LightGBM on TF-IDF features. **Run the baseline first.** It is 10 minutes of work and frequently wins on small datasets. If your transformer cannot beat it, you do not have enough labels yet — go label more instead of tuning hyperparameters.

### Step 5.3 — Retrieval fine-tuning

Once you have 200+ confirmed `(swing, true catalyst cluster)` pairs, fine-tune a bi-encoder with `sentence-transformers` using contrastive loss. Positive: the true catalyst. Hard negatives: other clusters from the same window.

This is the honest version of "the agent trains itself" — retrieval measurably improves at surfacing the right article over time, without anyone claiming to forecast anything.

**Baseline to beat:** the off-the-shelf embedding model.

**Use strict walk-forward splits.** Train on pairs from before date `D`, evaluate on swings after `D`. **Never shuffle.** Shuffled splits on time-series data leak future information and will make a broken model look excellent.

> ### ✅ Gate 5
> Each of the three components beats its baseline on a held-out, time-forward split. Any that doesn't, drop — the heuristic version is fine.

---

## PHASE 6 — Interface and operations

### Step 6.1 — Daily batch

After market close: scan the watchlist, run attribution on every swing with `|z| ≥ 2`, persist `Attribution` records to Postgres.

### Step 6.2 — Query interface

Natural language over stored attributions: *"why did NVDA move last Tuesday?"*, *"show me every unexplained swing this month."* Unexplained swings are genuinely worth reviewing — they mark gaps in your source coverage.

### Step 6.3 — Alerting

Push on `|z| ≥ 3`. **Make the verdict prominent** so `unexplained` is as visible as an explanation.

### Step 6.4 — Monitoring

Track per-source article counts per day, retrieval latency, and abstention rate over time. A sudden drop in abstention rate almost always means the timing filter broke — not that the world got more explicable.

**Track unexplained rate per ticker.** This is your coverage diagnostic, and it's the single most useful number in the whole dashboard.

```sql
SELECT ticker,
       count(*) FILTER (WHERE verdict = 'unexplained')::float / count(*) AS unexplained_rate,
       count(*) AS n
FROM attributions
WHERE created_at > now() - interval '90 days'
GROUP BY ticker ORDER BY unexplained_rate DESC;
```

If MU sits at 40% unexplained while NVDA is at 10%, that is not a model problem — it's a **source gap**. MU moves on memory-pricing news from trade press you aren't ingesting. The ticker at the top of that list tells you exactly which feed to add next.

Also break unexplained rate down by `swing_type`. A high rate on `gap` swings means your overnight window or your 8-K poller is broken, since gap moves should be your *easiest* cases.

---

# FAILURE MODES

| Failure | Symptom | Guard |
|---|---|---|
| Reactive journalism loop | Agent cites articles written *because* of the move | Hard timing filter (§2.1) |
| Syndication inflation | "Twelve sources confirm" for one wire story | Cluster-level `distinct_sources` (§2.2) |
| Sector move misattributed | Company headline blamed for sector selloff | Residual decomposition (§1.1) |
| Confabulation | Confident story where no catalyst exists | Code-level abstention (§3.2) + placebo test (§4.3) |
| Citation drift | Cited article doesn't support the claim | `validate_citations` node (§3.3) |
| Timezone corruption | Pre-move articles classified as post-move | UTC on write, assert on read (§0.4) |
| Silent feed death | A source stops appearing, nobody notices | Per-source daily counts (§6.4) |
| Adjusted price rewrite | Backtests change results month to month | Unadjusted bars + actions table (§0.2) |
| Shuffled train/test split | Model looks excellent, fails live | Walk-forward only (§5.3) |
| Following a stale tutorial | `AgentExecutor` deprecation warnings | LangGraph `StateGraph` only (§1.1) |
| **Onset ignored** | Post-move articles pass the pre-move filter; looks correct | Intraday onset detection (§1.3) |
| **News not backfillable** | Reach Phase 4 with 6 weeks of articles | Collector running from day one (Phase −1) |
| **Body-length bias** | WSJ never ranks; full-text sources dominate | Embed headline+summary only (§0.6) |
| **Circular annotation** | Recall@10 looks perfect, agent misses real catalysts | Blind annotation subset (§4.1) |
| **Unversioned records** | Can't tell if a metric moved from your edit or theirs | `prompt_version`, `model_id`, `config_hash` (§3.1b) |
| **Sector event, six explanations** | All six semis get separate stories for one SMH move | Sector ETFs as entities (§0.1b) |
| **Earnings blackout** | Silent on the highest-impact days of the year | `earnings_mode` branch (§1.7) |
| **Slow-burn stories missed** | 4-day bleed never fires a swing | Drift detector (§1.5) |

---

# TROUBLESHOOTING — THE FAILURES YOU'LL ACTUALLY HIT

Ordered by how likely they are to bite you, based on where this design has sharp edges.

## Gemini rejects your schema

**Symptom:** `with_structured_output(Attribution)` throws, or returns malformed output on complex swings while working on simple ones.

**Cause:** `Attribution → candidates[] → evidence[]` is three levels of nesting, right at Gemini's boundary. Gemini's supported JSON Schema subset is narrower than OpenAI's.

**Fix:** flatten per §3.1. Two flat lists joined on `candidate_id` in Python:

```python
class CandidateFlat(BaseModel):
    candidate_id: str
    catalyst: str
    event_type: Literal[...]
    direction_consistent: bool
    magnitude_plausible: bool
    confidence: Literal["high", "medium", "low"]

class EvidenceFlat(BaseModel):
    candidate_id: str          # foreign key
    cluster_id: int
    # ... rest unchanged

class AttributionFlat(BaseModel):
    # scalars unchanged
    candidates: list[CandidateFlat]
    evidence: list[EvidenceFlat]

def rehydrate(flat: AttributionFlat) -> Attribution:
    by_cand = defaultdict(list)
    for e in flat.evidence:
        by_cand[e.candidate_id].append(e)
    ...
```

Validation guarantees are identical. Do this during setup, not in Phase 4 — discovering it late means rerunning your eval sweeps.

## XBRL tags differ across your twelve companies

**Symptom:** revenue extraction works for four tickers and returns nothing for the rest.

**Cause:** no enforced tag standardization. Some file `Revenues`, others `RevenueFromContractWithCustomerExcludingAssessedTax`, others a custom extension tag.

**Fix:** inspect each company's `companyfacts` once and record the actual tags per ticker.

```python
def discover_tags(cik, headers):
    facts = fetch_companyfacts(cik, headers)["facts"]["us-gaap"]
    revenue_like = [k for k in facts if "Revenue" in k]
    return sorted(revenue_like, key=lambda k: -len(facts[k]["units"].get("USD", [])))
```

Take the tag with the most datapoints, eyeball it, write it into `config/watchlist.yaml` per ticker. One-time, ~30 minutes for twelve companies. Never assume one tag works across all of them.

## MinHash clustering is wrong once real wire copy arrives

**Symptom:** either "eight sources confirm" for one syndicated story (under-clustering), or genuinely distinct stories merged into one cluster (over-clustering).

**Cause:** Jaccard 0.9 and cosine 0.90 are starting points, not tuned values. Syndicated copy varies — some outlets run wire text verbatim, others add two paragraphs and a different headline.

**Fix:** once you have two weeks of real articles, dump the largest clusters and read them.

```sql
SELECT cluster_id, count(*), array_agg(DISTINCT source), min(headline)
FROM articles WHERE published_at > now() - interval '14 days'
GROUP BY cluster_id ORDER BY count(*) DESC LIMIT 20;
```

If clusters contain unrelated stories, raise the threshold. If the same Reuters story appears under three cluster IDs, lower it. **Tune on real data, not synthetic tests** — syndication patterns are what you're modeling and you can't fake them.

## EDGAR returns 403

**Cause:** missing or vague `User-Agent`. SEC requires a descriptive string with a contact email.

**Fix:** `{"User-Agent": "SwingAgent research you@example.com"}` on every request. If you're already blocked, wait it out and fix the header — retrying with the same header extends the block.

## EDGAR returns 404 for a valid company

**Cause:** CIK not zero-padded to 10 digits. Apple's CIK is `320193`; the URL needs `CIK0000320193`.

**Fix:** `str(cik).zfill(10)`, always. This is the single most common EDGAR bug and it looks like a missing company rather than a formatting error.

## pgvector similarity search returns poor matches

**Symptom:** semantically obvious matches don't surface; recall@10 fails Gate 2 despite the article being in the database.

**Cause:** you created the `ivfflat` index on an empty or nearly-empty table. ivfflat builds clusters from existing data — building it before loading gives you a useless partition.

**Fix:** load your articles first, *then* create the index. If you already created it, drop and rebuild:

```sql
DROP INDEX IF EXISTS articles_embedding_idx;
-- after data is loaded:
CREATE INDEX articles_embedding_idx ON articles
  USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

Rule of thumb for `lists`: rows/1000 for datasets under a million rows. Or use `hnsw` instead, which doesn't have this failure mode and is more forgiving at your scale.

## Gemini 429 during batch runs

**Symptom:** daily batch works for the first few tickers then fails.

**Cause:** free tier throttles by requests-per-minute as well as per-day. Firing attributions back to back trips RPM.

**Fix:** `tenacity` retry with exponential backoff on 429, plus a deliberate sleep between calls. Your batch has no latency requirement — a two-second gap costs nothing.

## `np.linalg.lstsq` returns garbage or NaN

**Symptom:** wild betas, NaN residuals, or wildly unstable coefficients day to day.

**Cause:** gaps in the bar history, or a ticker with insufficient history (SNDK).

**Fix:** assert the window is complete before fitting.

```python
assert len(y) == lookback, f"expected {lookback} bars, got {len(y)}"
assert not np.isnan(X).any(), "NaN in factor returns"
if history_days < 400:
    return None, "insufficient_history"
```

Fail loudly. A silently wrong beta produces plausible-looking residuals and corrupts everything downstream.

## Intraday bars unavailable for an older swing date

**Symptom:** onset detection works on recent swings, fails on historical ones.

**Cause:** free tiers limit intraday history to a short lookback window — daily bars go back years, 5-minute bars usually don't.

**Fix:** this is a real constraint, so plan for it. For historical swings without intraday data, set `swing_type = "unknown"` and fall back to a conservative 48-hour pre-move window. Flag those records — they're weaker evidence and shouldn't anchor your ranking-weight tuning. Going forward, **store intraday bars for every swing day as you detect it**, so the gap only affects your backfill.

## Timezone bugs that don't announce themselves

**Symptom:** none. That's the problem.

**Fix:** reject naive datetimes at the boundary.

```python
def assert_utc(ts):
    if ts.tzinfo is None:
        raise ValueError(f"naive datetime: {ts}")
    return ts.astimezone(timezone.utc)
```

Call it on every write and every read. RSS `pubDate`, Finnhub Unix timestamps, and SEC acceptance times all arrive in different formats.

## yfinance breaks after an upgrade

**Cause:** it's community-maintained and depends on undocumented Yahoo endpoints.

**Fix:** you already pinned it in `requirements.lock.txt`. Don't upgrade it casually. If it breaks anyway, Tiingo's free tier is your fallback for EOD history — worth registering for a key now rather than during an outage.

---

# WEEK ONE CHECKLIST

**Day one, before anything else:**

- [ ] `articles_raw` table created
- [ ] EDGAR 8-K poller running on 12 CIKs, every 10 min
- [ ] Company IR RSS feeds running, 12 URLs
- [ ] WSJ / CNBC / MarketWatch RSS running
- [ ] Dead-feed alert wired (zero articles in 24h → email)
- [ ] **Collector left running while you build everything below**

**Then:**

- [ ] Python 3.12 venv, `requirements.txt` installed, `requirements.lock.txt` written
- [ ] Postgres 16 + `pgvector`, schemas from §0.2 and §0.4 applied
- [ ] `torch` and `sentence-transformers` verified — embedding dim prints 768
- [ ] `langgraph.__version__` prints 1.x
- [ ] Gemini free-tier key working; setup step 7 returns a parsed object
- [ ] Real `Attribution` schema tested against Gemini — flatten if nesting fails (§3.1)
- [ ] `config/watchlist.yaml` with 12 tickers **plus 4 sector entities**
- [ ] Price backfill: 17 symbols, 2 years, daily bars
- [ ] **Intraday 5-min bars available for swing days** (needed for onset)
- [ ] SNDK history verified to start 2025-02-24, not spliced with pre-2016 SanDisk
- [ ] `analysis/decompose.py` implemented and plotted
- [ ] Onset detection verified by hand on 10 swings

That clears Gate 1. Everything downstream depends on the decomposition and onset being correct, so spend the time there before touching the LLM layer.

---

*This describes a research and analysis tool. It is not investment advice, and nothing it produces should be treated as a recommendation to buy or sell securities. Keep the system in alert-only mode until you have months of reviewed output.*