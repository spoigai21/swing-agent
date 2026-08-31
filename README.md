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

### The `swing` command

Installed globally (editable, so code edits take effect immediately):

```bash
uv tool install --editable . --python 3.12    # or: pipx install -e .
swing coverage        # what data do I have          (works, <100ms)
swing health          # per-source feed health       (works)
swing collect         # one collection pass          (works)
swing dbinit          # apply schema, idempotent     (works)
swing                 # interactive REPL
```

`swing why / stats / compare / unexplained / ask / batch` exist but report the
build step that unlocks them. **Only `ask` and `batch` ever call Gemini.**

### The collector

```bash
tail -f data/collector.log            # watch it
swing health                          # per-source counts + broken-feed check
kill $(cat data/collector.pid)        # stop
nohup caffeinate -is swing collect --daemon > data/collector.nohup.log 2>&1 &
  echo $! > data/collector.pid        # start  (see the sleep caveat below)
```

Polls: EDGAR 8-K every 10 min (12 CIKs) · IR RSS every 10 min · tier-3 press
every 15 min · dead-feed check every 6 h.

### ⚠️ Two persistence caveats — read both

**1. This Mac sleeps after 1 minute idle** (`pmset` reports `sleep 1`,
`powernap 0` on battery). `time.monotonic()` freezes across macOS sleep, so the
collector stops polling entirely — observed directly: one poll, then 61 minutes
with zero CPU and no polls. That is silent, permanent news loss.

The `caffeinate -is` wrapper above is the mitigation and is verified working,
but it keeps the Mac awake and does not survive a lid close. **The real fix is
an always-on host** — a $5 VPS or a Raspberry Pi running Postgres and the
collector. Every other part of this system is a batch job that can run anywhere;
only the collector must never stop.

**2. The collector runs under `nohup`. It survives closing the terminal,
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
pyproject.toml          entry point: swing = "swing.cli:main"
src/swing/
  cli.py                dispatcher + REPL
  commands.py           one function per subcommand
  paths.py              absolute path resolution, SWING_HOME override
  common/               settings, timeutil (UTC discipline), http, logging
  store/                schema, session, raw
  ingest/               collector, edgar, news_rss, health, config
config/                 watchlist.yaml (CIKs verified), sources.yaml
scripts/                bootstrap_db.py, com.swingagent.collector.plist
```

Package layout is `src/swing/` because a globally installed `swing` would
otherwise put `common` and `agent` on the system as top-level import names.

## Next: Step 1

Full `schema.sql`, SQLAlchemy models, alembic baseline, `store/queries.py`.
**Verify the Finnhub intraday candle endpoint first** (`CODEBASE-PLAN.md` §0.1 C1)
— onset detection depends on it and the plan's claim that it is free-tier looks
stale. Install the full `requirements.txt` at Step 2, not before; `torch` is a
~2 GB download and must not delay the clock.
