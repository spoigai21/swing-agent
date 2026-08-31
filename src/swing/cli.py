"""The `swing` command. Dispatcher + REPL.

Two modes, like `claude`: one-shot when you pass arguments, interactive REPL
when you don't. swing-cli.md Parts 5 and 6.

Startup budget: `swing coverage` must return in under 300ms, so nothing heavy
is imported at module level. Command implementations import their own
dependencies inside the function.
"""
from __future__ import annotations

import argparse
import sys

# Commands that call Gemini. Kept explicit so quota consumption is visible in
# the verb you typed. swing-cli.md Part 1.
LLM_COMMANDS = {"ask", "batch"}

HELP_TEXT = """
  /help              this text
  /coverage          what data do I have
  /tickers           list watchlist tickers
  /health            per-source feed health
  /quit              exit  (or Ctrl-D)

  <TICKER>           explain that ticker's most recent swing
  anything else      natural-language question (uses Gemini)
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swing", description="Stock swing attribution")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("coverage", help="what data do I have")

    c = sub.add_parser("collect", help="run the news collector")
    c.add_argument("--daemon", action="store_true", help="run continuously instead of one pass")

    sub.add_parser("health", help="per-source feed health")

    w = sub.add_parser("why", help="explain a swing")
    w.add_argument("ticker")
    w.add_argument("--date", help="YYYY-MM-DD; defaults to most recent swing")

    s = sub.add_parser("stats", help="aggregate swing stats")
    s.add_argument("ticker")
    s.add_argument("--days", type=int, default=90)

    cm = sub.add_parser("compare", help="side-by-side idio_share across tickers")
    cm.add_argument("tickers", nargs="+")
    cm.add_argument("--days", type=int, default=90)

    u = sub.add_parser("unexplained", help="swings with no catalyst found")
    u.add_argument("--ticker")
    u.add_argument("--days", type=int, default=30)

    a = sub.add_parser("ask", help="natural language (uses Gemini)")
    a.add_argument("question")

    b = sub.add_parser("batch", help="run the daily attribution batch")
    b.add_argument("--date")

    sub.add_parser("dbinit", help="apply the database schema (idempotent)")
    return p


def dispatch(args: argparse.Namespace) -> int:
    from swing import commands  # imported here to keep startup fast

    match args.cmd:
        case "coverage":
            return commands.coverage()
        case "collect":
            return commands.collect(daemon=args.daemon)
        case "health":
            return commands.health()
        case "dbinit":
            return commands.dbinit()
        case "why":
            return commands.why(args.ticker, args.date)
        case "stats":
            return commands.stats(args.ticker, args.days)
        case "compare":
            return commands.compare(args.tickers, args.days)
        case "unexplained":
            return commands.unexplained(args.ticker, args.days)
        case "ask":
            return commands.ask(args.question)
        case "batch":
            return commands.batch(args.date)
        case _:
            raise SystemExit(f"unknown command: {args.cmd}")


def repl() -> int:
    import atexit
    import readline  # stdlib: arrow-key history and line editing, no dependency
    from pathlib import Path

    from swing import commands

    history = Path.home() / ".swing_history"
    try:
        readline.read_history_file(history)
    except (FileNotFoundError, OSError):
        pass
    atexit.register(lambda: readline.write_history_file(history))

    cov = commands.coverage_data()
    tickers = set(cov["tickers"])
    print(
        f"  swing-agent · {len(tickers)} tickers · "
        f"{cov['articles']:,} articles from {cov['articles_from']}"
    )
    print("  Type a question, or /help for commands. Ctrl-D to exit.")

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        try:
            if line.startswith("/"):
                if _slash(line, commands) == "quit":
                    return 0
            elif line.upper() in tickers:
                # The most common thing you will do, so make it one word.
                commands.why(line.upper(), None)
            else:
                commands.ask(line)
        except Exception as e:  # noqa: BLE001 - a bad query must not kill the REPL  # noqa: BLE001
            print(f"error: {e}", file=sys.stderr)


def _slash(line: str, commands) -> str | None:
    cmd, *_rest = line[1:].split()
    match cmd:
        case "help":
            print(HELP_TEXT)
        case "coverage":
            commands.coverage()
        case "health":
            commands.health()
        case "tickers":
            print(", ".join(commands.coverage_data()["tickers"]))
        case "quit" | "exit" | "q":
            return "quit"
        case _:
            print(f"unknown command: /{cmd} — try /help")
    return None


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd is None:
        return repl()
    try:
        return dispatch(args)
    except KeyboardInterrupt:
        return 130
    except NotImplementedError as e:
        print(f"not built yet: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - report cleanly, never traceback at the user
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
