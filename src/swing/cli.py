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

# Commands that can call Gemini. Kept explicit so quota consumption is visible
# in the verb you typed. swing-cli.md Part 1.
LLM_COMMANDS = {"ask", "why", "batch", "daily", "placebo"}

HELP_TEXT = """
  Ask why a stock moved, in plain English:
    why is NVDA down?
    what happened to Tesla yesterday?
    why did MU jump on Aug 27?

  /stocks            the stocks I cover
  /health            is news collection working
  /coverage          how much news is stored
  /quit              exit  (or Ctrl-D)
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swing", description="Stock swing attribution")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("coverage", help="what data do I have")

    c = sub.add_parser("collect", help="run the news collector")
    c.add_argument("--daemon", action="store_true", help="run continuously instead of one pass")

    sub.add_parser("health", help="per-source feed health")

    w = sub.add_parser("why", help="why a stock moved (uses Gemini for unusual moves)")
    w.add_argument("ticker")
    w.add_argument("--date", help="YYYY-MM-DD; defaults to the latest completed session")

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
    b.add_argument("--limit", type=int, help="cap the run (free-tier daily quota)")

    bf = sub.add_parser("backfill", help="pull historical price bars")
    bf.add_argument("--years", type=int, default=2)
    bf.add_argument("--symbols", nargs="*")

    nb = sub.add_parser("backfill-news", help="historical company news (unblocks Gate 2)")
    nb.add_argument("--months", type=int, default=12)
    nb.add_argument("--tickers", nargs="*")

    be = sub.add_parser("backfill-events",
                        help="day-one history: SEC filings + GDELT world news")
    be.add_argument("--months", type=int, default=12,
                    help="how far back to pull GDELT (default 12)")
    be.add_argument("--tickers", nargs="*")
    be.add_argument("--no-gdelt", action="store_true",
                    help="SEC filings only; skip BigQuery entirely")

    nz = sub.add_parser("normalize", help="articles_raw -> articles (tier, tag, embed)")
    nz.add_argument("--limit", type=int, default=None, help="one batch of this size")

    sub.add_parser("prices", help="price coverage report + history-start checks")

    fa = sub.add_parser("factors", help="rebuild daily_factors (rolling decomposition)")
    fa.add_argument("--ticker")

    de = sub.add_parser("detect", help="detect swings on the residual")
    de.add_argument("--capture", action="store_true",
                    help="fetch intraday for each swing as it is found (slow)")

    ob = sub.add_parser("onsets", help="backfill intraday + onset for swings lacking it")
    ob.add_argument("--limit", type=int)

    rt = sub.add_parser("retrieve", help="build ranked clusters for swings")
    rt.add_argument("--swing", type=int, help="one swing id")
    rt.add_argument("--limit", type=int)

    an = sub.add_parser("annotate", help="hand-label swings with the true catalyst")
    an.add_argument("--blind", action="store_true",
                    help="hide retrieval until you commit (measures recall)")
    an.add_argument("--ticker")
    an.add_argument("--limit", type=int, default=10)
    an.add_argument("--progress", action="store_true")
    an.add_argument("--swing", type=int, help="annotate this swing id specifically")
    an.add_argument("--since", help="only swings on or after this date (YYYY-MM-DD)")

    sub.add_parser("metrics", help="the seven evaluation metrics")

    al = sub.add_parser("alert", help="notify on large moves (|z| >= 3)")
    al.add_argument("--days", type=int, default=3)
    al.add_argument("--min-z", type=float)
    al.add_argument("--dry-run", action="store_true")

    mo = sub.add_parser("monitor", help="operational dashboard")
    mo.add_argument("--days", type=int, default=90)

    dr = sub.add_parser("daily", help="the full post-close batch")
    dr.add_argument("--date")
    dr.add_argument("--limit", type=int, default=15,
                    help="max attributions (free-tier quota is 20/day)")
    dr.add_argument("--skip-prices", action="store_true")
    dr.add_argument("--skip-attribution", action="store_true")

    pl = sub.add_parser("placebo", help="confabulation test (uses Gemini quota)")
    pl.add_argument("--n", type=int, default=30)
    pl.add_argument("--seed", type=int, default=0)
    fw = sub.add_parser("fit-weights",
                        help="fit ranking weights on a time-forward split")
    fw.add_argument("--k", type=int, default=10,
                    help="recall@k to optimise (default 10)")
    rc = sub.add_parser("recheck-coverage",
                        help="re-check uncovered labels against the grown corpus")
    rc.add_argument("--link", help="<swing_id>=<article_id>[,<id>...] after you have "
                                   "read and confirmed the article")
    rc.add_argument("--min-similarity", type=float, default=0.35)
    sub.add_parser("source-precision",
                   help="which sources actually explain moves, measured")
    lg = sub.add_parser("label-gaps",
                        help="unlabelled swings likely to be an event type nothing has")
    lg.add_argument("--type", dest="etype",
                    help="only this event type (regulatory, litigation, …)")
    lg.add_argument("--limit", type=int, default=6)
    sub.add_parser("calibration",
                   help="is the agent's stated confidence worth anything")
    sub.add_parser("dbinit", help="apply the database schema (idempotent)")

    k = sub.add_parser("key", help="add, replace or test your Gemini API key")
    k.add_argument("value", nargs="?",
                   help="the key; omit it to be prompted with hidden input")
    k.add_argument("--check", action="store_true",
                   help="test the key currently in use (free, uses no quota)")
    k.add_argument("--no-verify", action="store_true",
                   help="save without checking it against Google first")

    it = sub.add_parser("init", help="first-time setup: config, keys, database")
    it.add_argument("--home", help="where to put config and data (default ~/.swing)")
    it.add_argument("--force", action="store_true",
                    help="overwrite existing config files with the shipped defaults")
    it.add_argument("--non-interactive", action="store_true",
                    help="take defaults and existing .env values, ask nothing")
    return p


#: `init` is what you run when nothing is set up, so it must not be gated on
#: being set up. Everything else is, at this one choke point rather than in
#: twenty command bodies.
NEEDS_NO_CONFIG = {"init"}


def dispatch(args: argparse.Namespace) -> int:
    from swing import commands  # imported here to keep startup fast

    if args.cmd not in NEEDS_NO_CONFIG:
        from swing.paths import require_initialised

        require_initialised()

    match args.cmd:
        case "key":
            from swing.interface import setup

            return setup.key_command(args.value, check=args.check,
                                     verify=not args.no_verify)
        case "init":
            from pathlib import Path

            from swing.interface import setup

            return setup.run(Path(args.home) if args.home else None,
                             force=args.force,
                             interactive=not args.non_interactive)
        case "coverage":
            return commands.coverage()
        case "collect":
            return commands.collect(daemon=args.daemon)
        case "health":
            return commands.health()
        case "fit-weights":
            from swing.models.weights import report as weights_report

            print(weights_report(k=args.k))
            return 0
        case "recheck-coverage":
            from swing.eval.recheck import main as recheck_main

            return recheck_main(args.link, args.min_similarity)
        case "source-precision":
            from swing.eval.sources import main as sources_main

            return sources_main()
        case "label-gaps":
            from swing.eval.classgaps import main as gaps_main

            return gaps_main(args.limit, args.etype)
        case "calibration":
            from swing.eval.calibration import main as calibration_main

            return calibration_main()
        case "dbinit":
            return commands.dbinit()
        case "backfill":
            return commands.backfill(args.years, args.symbols)
        case "backfill-events":
            return commands.backfill_events(args.months, args.tickers, args.no_gdelt)
        case "backfill-news":
            return commands.backfill_news(args.months, args.tickers)
        case "normalize":
            return commands.normalize(args.limit)
        case "prices":
            return commands.prices()
        case "factors":
            return commands.factors(args.ticker)
        case "detect":
            return commands.detect(args.capture)
        case "onsets":
            return commands.onsets(args.limit)
        case "retrieve":
            return commands.retrieve(args.swing, args.limit)
        case "annotate":
            return commands.annotate(args.blind, args.ticker, args.limit,
                                     args.progress, args.swing, args.since)
        case "metrics":
            return commands.metrics()
        case "alert":
            return commands.alert(args.days, args.min_z, args.dry_run)
        case "monitor":
            return commands.monitor(args.days)
        case "daily":
            return commands.daily(args.date, args.limit, args.skip_prices,
                                  args.skip_attribution)
        case "placebo":
            return commands.placebo(args.n, args.seed)
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
            return commands.batch(args.date, args.limit)
        case _:
            raise SystemExit(f"unknown command: {args.cmd}")


def repl() -> int:
    import atexit
    import readline  # stdlib: arrow-key history and line editing, no dependency
    import threading
    from pathlib import Path

    from swing import commands
    from swing.common import logging as log
    from swing.ingest.config import stocks
    from swing.paths import DATA

    log.setup(logfile=DATA / "swing.log", console=False)
    history = Path.home() / ".swing_history"
    try:
        readline.read_history_file(history)
    except (FileNotFoundError, OSError):
        pass
    atexit.register(lambda: readline.write_history_file(history))
    # Load the embedding model while the user types, so the first answer is quick.
    threading.Thread(target=_warm_up, daemon=True).start()

    print("  swing · ask why a stock went up or down")
    print(f"  stocks: {' '.join(sorted(stocks()))}")
    print("  e.g. why is NVDA down?   ·   /help   ·   Ctrl-D to exit")

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
            else:
                # Every line is a question; a bare ticker ("NVDA") works too.
                commands.ask(line)
        except KeyboardInterrupt:
            print("\n  (cancelled)")
        except Exception as e:  # noqa: BLE001 - a bad question must not kill the REPL
            print(f"error: {e}", file=sys.stderr)


def _warm_up() -> None:
    import logging

    try:
        import swing.interface.explain  # noqa: F401 - quiets library output before loading
        from swing.ingest.normalize import _embed_texts

        _embed_texts(["warm up"])
    except Exception:  # warming is an optimisation, never an error
        logging.getLogger("cli").debug("embedding warm-up failed", exc_info=True)


def _slash(line: str, commands) -> str | None:
    cmd, *_rest = line[1:].split()
    match cmd:
        case "help":
            print(HELP_TEXT)
        case "coverage":
            commands.coverage()
        case "health":
            commands.health()
        case "stocks" | "tickers":
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
