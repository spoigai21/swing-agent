# Stock Swing Attribution Agent

Detects unusual price moves, retrieves news across sources, and produces an
evidence-backed explanation of *why* a stock moved — or states that no catalyst
can be identified.

Specs: `agent-plan.md` (what and why) · `data-sources.md` (APIs) ·
`CODEBASE-PLAN.md` (module layout, schema, build order).

**Status: Step 0 complete — the news collector is running.**
Everything else is unbuilt by design; news does not backfill, so the collector
starts before the rest of the system exists.

---

## Runbook

### The collector

```bash
tail -f data/collector.log            # watch it
.venv/bin/python ingest/collector.py --report   # per-source counts
kill $(cat data/collector.pid)        # stop
nohup .venv/bin/python ingest/collector.py > data/collector.nohup.log 2>&1 &
  echo $! > data/collector.pid        # start
```

Polls: EDGAR 8-K every 10 min (12 CIKs) · IR RSS every 10 min · tier-3 press
every 15 min · dead-feed check every 6 h.

### ⚠️ Persistence caveat — read this

The collector currently runs under `nohup`. **It survives closing the terminal,
but not a reboot or logout.** A macOS LaunchAgent was tried and does not work
from this location: `~/Desktop` is TCC-protected, and a launchd background agent
does not inherit Full Disk Access, so the Python interpreter blocks forever in
`_PyConfig_InitPathConfig → open()` before any project code runs.

Two durable fixes, either one is a few minutes:

1. **Move the repo out of `~/Desktop`** (e.g. `~/projects/swing-agent`), recreate
   the venv there, then install `scripts/com.swingagent.collector.plist`:
   ```bash
   sed "s|__REPO__|$PWD|g" scripts/com.swingagent.collector.plist \
     > ~/Library/LaunchAgents/com.swingagent.collector.plist
   launchctl load ~/Library/LaunchAgents/com.swingagent.collector.plist
   ```
2. **Grant Full Disk Access** to the interpreter in
   System Settings → Privacy & Security → Full Disk Access, then load the plist.

Until then: after any reboot, restart the collector with the command above.
**Check `--report` shows growth every few days.** A silently dead feed for three
weeks is three weeks of unrecoverable data.

### Database

```bash
docker compose up -d                  # Postgres 16 + pgvector on :5433
docker compose ps
.venv/bin/python scripts/bootstrap_db.py   # idempotent schema apply
docker exec -e PGPASSWORD=swing swing-db psql -U swing -d swing_agent
```

---

## What exists

```
common/     settings, timeutil (UTC discipline), http (rate limits), logging
store/      schema_phase_minus1.sql, session.py, raw.py
ingest/     collector.py (daemon), edgar.py, news_rss.py, health.py, config.py
config/     watchlist.yaml (12 stocks + 4 sectors, CIKs verified), sources.yaml
scripts/    bootstrap_db.py, com.swingagent.collector.plist
```

## Next: Step 1

Full `schema.sql`, SQLAlchemy models, alembic baseline, `store/queries.py`.
**Verify the Finnhub intraday candle endpoint first** (`CODEBASE-PLAN.md` §0.1 C1)
— onset detection depends on it and the plan's claim that it is free-tier looks
stale. Install the full `requirements.txt` at Step 2, not before; `torch` is a
~2 GB download and must not delay the clock.
