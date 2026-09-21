# swing

**Why did that stock move?** `swing` detects unusual price moves, finds the news
published *before* they started, and explains them — or tells you it can't.

```console
$ swing why NVDA --date 2025-10-06

NVDA fell 1.1% on Mon Oct 6.
  Split: market +0.6% · chip stocks +1.3% · NVDA on its own -3.1%

Why: AMD announced a major strategic partnership with OpenAI to deploy
6 gigawatts of GPUs, presenting a substantial competitive threat to
Nvidia's AI chip market dominance.
  • Mon Oct 6, 7:04am ET · SEC filing · AMD 8-K — Item 1.01,3.02,7.01,9.01
    — AMD AND OPENAI ANNOUNCE STRATEGIC PARTNERSHIP TO DEPLOY 6 GIGAWATTS
    https://www.sec.gov/Archives/edgar/data/2488/000119312525230895/d28189d8k.htm
  • Mon Oct 6, 7:30am ET · CNBC · OpenAI looks to take 10% stake in AMD
    https://www.cnbc.com/2025/10/06/openai-amd-chip-deal-ai.html
  • Mon Oct 6, 8:15am ET · Theverge · AMD teams up with OpenAI to challenge
    Nvidia's AI chip dominance
    https://www.theverge.com/news/792650/amd-openai-five-year-ai-chip-agreement
  Confidence: high
```

Note the split. NVDA closed down only 1.1%, which on its own looks like noise —
but the market was up and chip stocks were up 1.3%, so the company's own move
was **-3.1%**. That is the move worth explaining, and the explanation came from
a **competitor's** SEC filing, timestamped 2h26m before the drop began.


---

## The one thing to know

`swing` abstains. When nothing published before a move explains it, it says
`unexplained` instead of assembling a plausible story from whatever is nearby.

In 38 tests where it was handed deliberately unrelated evidence, it invented a
cause **0 times**. That is the property the rest of the design exists to
protect, and it is measured on every release rather than asserted.

---

## How it works

```
  ┌─ collect ──────────────────────────────────────────────────────────┐
  │  SEC EDGAR · GDELT (Reuters/Bloomberg/WSJ) · company IR feeds ·    │
  │  CNBC · MarketWatch · analyst ratings · press wires                │
  └────────────────────────────┬───────────────────────────────────────┘
                               ▼
  1. DECOMPOSE   Split the day's return into market + sector + the part
                 that is this company alone. A stock up 3% on a day the
                 market is up 3% has not done anything.

  2. DETECT      Flag only moves where the idiosyncratic residual is
                 statistically unusual for that ticker.

  3. TIME        Find the minute the move actually started, from intraday
                 bars. This is what makes "before" mean something.

  4. RETRIEVE    Cluster news around that moment and rank it. Everything
                 is labelled pre_move or post_move against the onset.

  5. EXPLAIN     Ask the model to pick a catalyst from the pre-move
                 evidence, citing specific articles — or to abstain.

  6. GUARD       Drop any citation that does not resolve to a real
                 retrieved article, then re-check the answer still stands.
                 A verdict resting on a dropped citation becomes
                 `unexplained`.
```

Steps 1-4 use no model at all. The model only writes the final judgement, and
it can only cite what step 4 actually retrieved.

---

## Install

Needs **Python 3.12**.

```bash
pip install swing-agent        # or: uv tool install swing-agent
swing init                     # asks for your keys, sets up the database
```

The install is about 1 GB, mostly PyTorch for the embedding model that ranks
news — more on Linux, where PyTorch's default build bundles GPU libraries. That
model runs on your machine. Gemini only ever sees the price move and the
headlines (plus a summary of up to 300 characters each) of the news it is asked
to judge.

`swing init` walks you through setup and writes everything to `~/.swing/.env`.

You need **Postgres with pgvector**. If you don't have one:

```bash
docker run -d --name swing-db -p 5433:5432 \
  -e POSTGRES_USER=swing -e POSTGRES_PASSWORD=swing -e POSTGRES_DB=swing_agent \
  pgvector/pgvector:pg16
```

Then load history and start collecting:

```bash
swing backfill                 # price bars
swing backfill-events          # SEC filings + world news, years back
swing collect --daemon &       # keep running — see below
swing why NVDA
```

### Your Gemini API key

**swing works with Google Gemini only.** OpenAI, Anthropic and other providers'
keys will not work. Every number in this README was measured on
`gemini-3.6-flash`, and the free tier is enough to use it daily.

1. Get a free key at **[aistudio.google.com/apikey](https://aistudio.google.com/apikey)**
   — sign in with a Google account and click *Create API key*. It starts with `AIza`.
2. Give it to swing:

   ```bash
   swing key                      # paste when asked; typing is hidden
   ```

That's it. The key is checked with Google before it is saved, so a typo is
caught immediately, and checking it does **not** use any of your daily requests.

```bash
swing key --check                  # is my key still working?
swing key                          # replace it
```

The free tier allows **20 explanations a day** per Google Cloud project — not
per key, so a second key from the same project adds nothing. Only moves that
are actually unusual use a request; quiet days, charts and everything else are
free. When the day's 20 are gone, `swing` says so and resets at midnight Pacific.

If you already export `GEMINI_API_KEY` in your shell, that copy takes priority
over the saved one — `swing key` will warn you if they differ.

### Other keys

| Key | Needed for | Free? |
|---|---|---|
| `SEC_USER_AGENT` | SEC filings. Any string with your email in it. | yes |
| `DATABASE_URL` | everything | yes |
| `FINNHUB_API_KEY` | company news, analyst ratings | optional, free tier |
| `GOOGLE_CLOUD_PROJECT` | GDELT wire coverage via BigQuery | optional, 1 TB/month free |

`swing init` asks for the first two. Edit `~/.swing/.env` for the rest.

### ⚠️ Keep the collector running

News cannot be backfilled. An RSS feed serves the last 20-50 items and free API
tiers cap how far back you can ask, so **every hour the collector is not running
is coverage that is gone permanently.** `swing backfill-events` recovers SEC
filings and GDELT because those two have real archives; nothing else does.

If your machine sleeps, the collector stops. Run it somewhere that stays awake.

---

## Commands

**Asking things**

```bash
swing                          # interactive prompt
swing why NVDA                 # latest session
swing why TSLA --date 2026-09-08
swing ask "what is unexplained recently"
swing unexplained              # moves it could not account for
swing stats NVDA --days 90     # beta, R², residual vol, swing frequency
swing compare NVDA AVGO MRVL   # idiosyncratic share side by side
```

**Running it**

```bash
swing init                     # first-time setup
swing key                      # add or replace your Gemini API key
swing collect --daemon         # the news collector
swing health                   # per-source freshness, broken feeds
swing monitor                  # operational dashboard
swing daily                    # post-close batch + alerts
swing metrics                  # the evaluation numbers below
```

Everything else (`annotate`, `placebo`, `retrieve`, `factors`, …) is pipeline
internals and evaluation tooling.

---

## What it is honest about

`swing metrics` prints these against live data. They are measurements, not
targets:

| Metric | Meaning |
|---|---|
| **confabulation** | Given unrelated evidence, how often does it invent a cause? Must be near zero. |
| **catalyst coverage** | Of moves with a knowable cause, how often is the news actually in the corpus? Bounded by your sources, not the model. |
| **attribution accuracy** | When it does explain, is the explanation right? |
| **abstention precision** | When it says `unexplained`, was there really nothing? |
| **citation validity** | Does every cited article exist? Must be exactly 1.00. |

Coverage is the honest weak spot. It runs around **0.70** on a well-fed
install — roughly three in ten moves that *had* a findable cause still come back
`unexplained`, because the story was never collected. More sources and a longer
running collector move that number; nothing else does.

---

## Configuration

`swing init` writes to `~/.swing/`:

```
~/.swing/
  .env                 keys (0600)
  config/
    watchlist.yaml     which tickers, their CIKs and name aliases
    sources.yaml       feeds, their tiers and poll intervals
    thresholds.yaml    what counts as an unusual move
    prompts/           the attribution prompts, versioned
  data/                logs and local state
```

Edit them freely — `swing init` never overwrites your changes. Set `SWING_HOME`
to keep them somewhere else.

---

## Development

```bash
git clone https://github.com/spoigai21/swing-agent && cd swing-agent
uv sync --extra dev
docker compose up -d
uv run pytest -q
```

A checkout uses the repo's own `config/` directory, so it never touches
`~/.swing`. Design notes live in `agent-plan.md`, `data-sources.md` and
`CODEBASE-PLAN.md`.

---

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

**This is not investment advice.** It is a research tool that reports what the
news said before a price move. It does not predict prices, recommend trades, or
know anything about your situation.

`unexplained` means *no catalyst was found in the sources collected*, which is
not the same as *no catalyst existed*. An explanation it does give is a starting
point for your own reading, not a conclusion — verify anything you act on
against the linked source.
