# Building the `swing` Command

> **Status (2026-09-14): this is the original spec. What was built differs in these places:**
> - **The product is one question.** `swing` opens a prompt; every line is a question
>   ("why is NVDA down?") answered by `interface/explain.py`, which fetches fresh prices
>   and news first. A bare ticker works too. `swing ask "..."` and `swing why NVDA` use
>   the same path.
> - **Gemini use:** `ask` and `why` call Gemini only when the stock's own move is unusual
>   (one request); `batch`, `daily` and `placebo` also call it. Normal days are answered
>   with no model call.
> - **`insight-agent-guide.md` never existed.** The forecast refusal was specified from
>   scratch (`interface/guardrail.py`), and `idio_share` is `var(residual)/var(return)`.
> - **Package layout** is `src/swing/` (`swing = "swing.cli:main"`), not `agent/cli.py`;
>   see CODEBASE-PLAN.md §8.
> - Design decisions and findings live in CODEBASE-PLAN.md §16.

How to turn the insight agent into a terminal command you invoke by name — type `swing` from anywhere and it runs, the way `claude` does.

Companion to `swing-attribution-agent-plan.md` and `insight-agent-guide.md`. This covers packaging and the command surface only; the tools and guardrails are specified in the insight guide.

---

## What you're building

```bash
$ swing coverage
articles from 2026-08-30 · 1,284 articles · 6 sources · 0 attributions

$ swing why NVDA
NVDA fell 2.40 on 2026-08-28...

$ swing
  swing-agent · 12 tickers · articles from 2026-08-30
  Type a question, or /help. Ctrl-D to exit.

>
```

Two modes, same as `claude`: **one-shot** when you pass arguments, **interactive REPL** when you don't.

---

# PART 1 — COMMAND SURFACE

Design this before writing code. Adding subcommands later is easy; renaming them after you've built muscle memory is not.

| Command | LLM? | Does |
|---|---|---|
| `swing` | — | Interactive REPL |
| `swing coverage` | No | What data do I have? |
| `swing why NVDA` | No | Most recent swing for a ticker |
| `swing why NVDA --date 2026-08-14` | No | One specific swing |
| `swing stats MU --days 90` | No | Aggregate swing stats |
| `swing compare NVDA MRVL MU` | No | Side-by-side `idio_share` |
| `swing unexplained --days 30` | No | Swings with no catalyst found |
| `swing ask "what drove MU?"` | **Yes** | Natural language |
| `swing collect` | No | Run the collector once |
| `swing batch` | No | Run the daily attribution batch |
| `swing batch --date 2026-08-14` | Yes | Re-run one day |

**Only `ask` and `batch` touch Gemini.** Everything else is SQL. Keeping that split visible in the command surface means you can see your quota consumption in the verb you typed.

### Naming

`swing` is short, unclaimed, and unambiguous. Check nothing else owns it first:

```bash
which swing || echo "free"
```

---

# PART 2 — PACKAGE STRUCTURE

An installable command requires a real Python package with an entry point. Three files change.

```
swing-agent/
├── pyproject.toml          # ← entry point declared here
├── agent/
│   ├── __init__.py
│   ├── cli.py              # ← new: dispatcher + REPL
│   ├── insight_tools.py
│   ├── graph.py
│   └── ...
```

### `pyproject.toml`

```toml
[project]
name = "swing-agent"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "langgraph>=1.0",
    "langchain>=1.0",
    "langchain-google-genai>=2.0",
    "psycopg[binary]>=3.2",
    "pgvector>=0.3",
    "sentence-transformers>=3.0",
    "feedparser>=6.0",
    "httpx>=0.27",
    "tenacity>=8.2",
    "pydantic>=2.7",
    "python-dotenv>=1.0",
    "pyyaml>=6.0",
    "numpy>=1.26,<3",
    "pandas>=2.2",
]

[project.scripts]
swing = "agent.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

The `[project.scripts]` line is the whole trick. On install, pip generates an executable named `swing` that calls `main()` in `agent/cli.py`.

**Keep `requirements.txt` too.** `pyproject.toml` declares loose deps for the package; `requirements.lock.txt` pins what you actually resolved. They serve different purposes.

---

# PART 3 — INSTALLATION

Three ways. They differ in whether the command works outside your project directory.

### Option A — venv editable install (simplest)

```bash
source .venv/bin/activate
pip install -e .
swing coverage
```

Code edits take effect immediately, no reinstall. **But `swing` only exists while the venv is active.** Fine if you always work inside the project.

### Option B — pipx (closest to how `claude` behaves) ⭐

```bash
brew install pipx        # or: python -m pip install --user pipx
pipx ensurepath          # adds ~/.local/bin to PATH, restart shell after
pipx install -e /full/path/to/swing-agent
```

Isolated environment, globally available, works from any directory with no venv activation. This is what you want.

Note pipx installs deps into its own env, separate from your `.venv`. Your dev venv still exists for running tests and scripts.

### Option C — shell wrapper (zero packaging)

If pipx gives you trouble, this works and takes 30 seconds:

```bash
mkdir -p ~/.local/bin
cat > ~/.local/bin/swing << 'EOF'
#!/usr/bin/env bash
exec /full/path/to/swing-agent/.venv/bin/python -m agent.cli "$@"
EOF
chmod +x ~/.local/bin/swing
```

Ensure `~/.local/bin` is on your `PATH`. Crude, completely reliable, and it uses your existing venv.

**Recommendation: B, falling back to C.** Skip A unless you're only ever in the project directory.

---

# PART 4 — ⚠️ THE WORKING-DIRECTORY PROBLEM

**This is the bug you will hit, and it will be confusing.**

Once `swing` is global, you'll run it from `~/Documents` or anywhere else. Every relative path in your code then resolves against the wrong directory:

```python
load_dotenv()                              # looks for ./.env — not found
yaml.safe_load(open("config/watchlist.yaml"))   # FileNotFoundError
```

The symptom is misleading: `swing` works perfectly inside the project and fails with a missing-key or missing-file error everywhere else, so it looks like an install problem when it's a path problem.

### Fix: resolve paths against a project root, not the cwd

```python
# agent/paths.py
import os
from pathlib import Path

def project_root() -> Path:
    """SWING_HOME if set, else the directory containing the agent package."""
    if env := os.getenv("SWING_HOME"):
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent.parent

ROOT      = project_root()
CONFIG    = ROOT / "config"
ENV_FILE  = ROOT / ".env"
WATCHLIST = CONFIG / "watchlist.yaml"
CIK_MAP   = CONFIG / "cik_map.json"
```

Then everywhere else:

```python
from dotenv import load_dotenv
from agent.paths import ENV_FILE, WATCHLIST

load_dotenv(ENV_FILE)                      # absolute, works from anywhere
watchlist = yaml.safe_load(WATCHLIST.read_text())
```

**Never use a bare relative path anywhere in the codebase.** Add `SWING_HOME` to your shell profile as a belt-and-braces measure:

```bash
echo 'export SWING_HOME="$HOME/code/swing-agent"' >> ~/.zshrc
```

This also applies to `collector.py` once it runs under cron — cron's working directory is not your project. Same fix.

---

# PART 5 — THE DISPATCHER

`agent/cli.py`. Argparse subparsers; no new dependency.

```python
import argparse
import sys

from dotenv import load_dotenv
from agent.paths import ENV_FILE

load_dotenv(ENV_FILE)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swing", description="Stock swing attribution")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("coverage", help="what data do I have")

    w = sub.add_parser("why", help="explain a swing")
    w.add_argument("ticker")
    w.add_argument("--date", help="YYYY-MM-DD; defaults to most recent swing")

    s = sub.add_parser("stats", help="aggregate swing stats")
    s.add_argument("ticker")
    s.add_argument("--days", type=int, default=90)

    c = sub.add_parser("compare", help="side-by-side across tickers")
    c.add_argument("tickers", nargs="+")
    c.add_argument("--days", type=int, default=90)

    u = sub.add_parser("unexplained", help="swings with no catalyst found")
    u.add_argument("--ticker")
    u.add_argument("--days", type=int, default=30)

    a = sub.add_parser("ask", help="natural language (uses Gemini)")
    a.add_argument("question")

    sub.add_parser("collect", help="run the collector once")

    b = sub.add_parser("batch", help="run the daily attribution batch")
    b.add_argument("--date")

    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.cmd is None:              # bare `swing` → REPL
        return repl()

    try:
        return dispatch(args)
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def dispatch(args) -> int:
    from agent import commands          # import here: keeps startup fast
    return {
        "coverage":    lambda: commands.coverage(),
        "why":         lambda: commands.why(args.ticker, args.date),
        "stats":       lambda: commands.stats(args.ticker, args.days),
        "compare":     lambda: commands.compare(args.tickers, args.days),
        "unexplained": lambda: commands.unexplained(args.ticker, args.days),
        "ask":         lambda: commands.ask(args.question),
        "collect":     lambda: commands.collect(),
        "batch":       lambda: commands.batch(args.date),
    }[args.cmd]()


if __name__ == "__main__":
    sys.exit(main())
```

### Keep startup fast

Import heavy modules **inside** the command functions, never at module top level. `sentence-transformers` alone adds seconds to import. If `swing coverage` — a single SQL query — takes four seconds because torch loaded, you'll stop using it.

```python
def ask(question: str):
    from agent.insight import insight_agent    # heavy, only when needed
    ...
```

Target: `swing coverage` returns in under 300ms.

---

# PART 6 — THE REPL

Bare `swing` drops into a prompt loop.

```python
import readline          # stdlib; gives arrow-key history for free on Unix
import atexit
from pathlib import Path

HISTORY = Path.home() / ".swing_history"

BANNER = """  swing-agent · {n} tickers · articles from {since}
  Type a question, or /help for commands. Ctrl-D to exit.
"""

def repl() -> int:
    from agent import commands

    try:
        readline.read_history_file(HISTORY)
    except FileNotFoundError:
        pass
    atexit.register(readline.write_history_file, HISTORY)

    cov = commands.coverage_data()
    print(BANNER.format(n=len(cov["tickers"]), since=cov["articles_from"]))

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue

        if line.startswith("/"):
            if handle_slash(line, commands) == "quit":
                return 0
            continue

        # Bare word that's a known ticker → treat as `why`
        if line.upper() in cov["tickers"]:
            commands.why(line.upper(), None)
            continue

        commands.ask(line)          # everything else → natural language


def handle_slash(line: str, commands) -> str | None:
    cmd, *rest = line[1:].split()
    match cmd:
        case "help":     print(HELP_TEXT)
        case "coverage": commands.coverage()
        case "tickers":  print(", ".join(commands.coverage_data()["tickers"]))
        case "compare":  commands.compare(rest or commands.semis(), 90)
        case "quit" | "exit" | "q":
            return "quit"
        case _:
            print(f"unknown command: /{cmd} — try /help")
    return None
```

`import readline` is the whole trick for arrow-key history and line editing. Stdlib, Unix, no dependency.

**Three REPL affordances worth having:**

- **Bare ticker shortcut.** Typing `NVDA` runs `why NVDA`. It's the most common thing you'll do.
- **Slash commands** for the zero-LLM queries, so the free path stays reachable inside the REPL.
- **Persistent history** at `~/.swing_history`, so yesterday's questions survive.

**One thing not to add: conversation memory.** Each REPL line is an independent query. This is deliberate — see the insight guide. Carrying context between turns lets a small-sample warning from one question silently shape the answer to another, and you won't see it happen.

---

# PART 7 — SHELL COMPLETION (OPTIONAL)

Tab-completion for subcommands and tickers. Nice, not necessary.

```bash
pip install argcomplete
activate-global-python-argcomplete --user
```

Then add to `cli.py`:

```python
import argcomplete

def main() -> int:
    parser = build_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
```

Skip this until the command surface has stopped changing.

---

# PART 8 — VERIFICATION

Run these **from your home directory**, not the project. That's the whole point.

```bash
cd ~
swing coverage                      # data summary, no error
swing why NVDA                      # most recent NVDA swing
swing stats MU --days 90            # aggregates with n
swing compare NVDA MRVL MU          # side-by-side
swing unexplained --days 30         # no-catalyst swings
swing ask "what drove MU last month?"   # only this hits Gemini
swing                               # REPL, banner shows, Ctrl-D exits
```

- [ ] Every command works from `~`, not just the project directory
- [ ] `swing coverage` returns in under 300ms
- [ ] `swing --help` lists all subcommands
- [ ] REPL history survives a restart
- [ ] Only `ask` and `batch` make network calls — verify with logging
- [ ] `swing ask "is NVDA a buy?"` refuses before calling Gemini

That last one matters. The forecast guardrail from the insight guide must fire in the CLI path too, not just wherever you first implemented it.

---

# TROUBLESHOOTING

**`swing: command not found` after pipx install**
`pipx ensurepath`, then restart your shell. Confirm `~/.local/bin` is on `PATH` with `echo $PATH`.

**Works in project directory, fails everywhere else**
The Part 4 path problem. Search for relative paths: `grep -rn "open('config\|open(\"config\|load_dotenv()" agent/`.

**`swing` is slow to start**
Heavy imports at module level. Move `torch`, `sentence-transformers`, and `langchain` imports inside the functions that need them.

**Code edits don't take effect**
You installed non-editable. Reinstall with `-e`: `pipx install -e /path/to/swing-agent --force`.

**`ModuleNotFoundError: agent` under pipx**
Missing `agent/__init__.py`, or `[tool.setuptools]` isn't finding the package. Add explicitly:

```toml
[tool.setuptools]
packages = ["agent"]
```

**Cron runs of `collector.py` fail while manual runs work**
Same path problem as Part 4, plus cron has a minimal `PATH` and no shell profile. Use absolute paths and set `SWING_HOME` in the crontab:

```
SWING_HOME=/Users/you/code/swing-agent
*/10 * * * * /Users/you/.local/bin/swing collect >> /Users/you/code/swing-agent/collector.log 2>&1
```

Once `swing collect` exists, use it in cron instead of calling `collector.py` directly — one entry point, one place for path resolution.

---

# BUILD ORDER

1. `pyproject.toml` with the `[project.scripts]` entry point
2. `agent/paths.py` — do this before anything else touches a file
3. `agent/cli.py` with `coverage` only, then `pipx install -e .`
4. Verify `swing coverage` works from `~`
5. Add remaining subcommands one at a time
6. REPL last

Step 4 is the real gate. If `coverage` works from your home directory, the packaging and path handling are both correct and everything after is just adding subcommands.