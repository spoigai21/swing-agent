"""`swing init` — turn a fresh install into a working one.

The 0.1.0 release was `pip install`-able and then immediately dead: the config
lives outside the package, so a new user got "cannot locate the project root"
from the first command they typed. This creates the home directory that
`swing.paths` looks for, seeds it from the config shipped in the wheel, and
collects the two credentials nothing works without.

Deliberately NOT a full setup: it does not install Postgres and does not start
the collector. It tells you what to run next, because a wizard that silently
launches background daemons is worse than one that does not.
"""
from __future__ import annotations

import shutil
import stat
from pathlib import Path

from swing.paths import PACKAGED_DEFAULTS

# The docker-compose in the repo publishes Postgres here, so it is the right
# guess for anyone following the README rather than bringing their own database.
DEFAULT_DATABASE_URL = "postgresql://swing:swing@localhost:5433/swing_agent"

# Asked in order. (env var, prompt, default, required)
PROMPTS: tuple[tuple[str, str, str, bool], ...] = (
    ("SEC_USER_AGENT",
     ("A contact string for the SEC. They return 403 without one, and may block\n"
      "you rather than get in touch if it has no email.\n"
      "  e.g. 'Jane Smith jane@example.com'"),
     "", True),
    ("DATABASE_URL",
     ("Postgres connection string. `docker compose up -d` in the repo serves\n"
      "the default below."),
     DEFAULT_DATABASE_URL, True),
    ("GEMINI_API_KEY",
     ("Gemini API key, for writing the explanations — free at\n"
      "https://aistudio.google.com/apikey  (free tier: 20 requests/day).\n"
      "Leave blank to set it later; everything but `why` and `ask` still works."),
     "", False),
    ("FINNHUB_API_KEY",
     ("Finnhub key, optional — adds company news and analyst ratings.\n"
      "Free at https://finnhub.io/register"),
     "", False),
)


def seed_config(home: Path, *, force: bool = False) -> list[str]:
    """Copy the packaged defaults into `home/config`, never overwriting edits."""
    dest = home / "config"
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for src in sorted(PACKAGED_DEFAULTS.rglob("*")):
        if src.is_dir():
            continue
        target = dest / src.relative_to(PACKAGED_DEFAULTS)
        if target.exists() and not force:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        written.append(str(target.relative_to(home)))
    return written


def read_env(path: Path) -> dict[str, str]:
    """Existing values, so re-running init does not wipe a working setup."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def write_env(path: Path, values: dict[str, str]) -> None:
    """⚠️ 0600. This file holds API keys and a database password."""
    body = "\n".join(f"{k}={v}" for k, v in values.items() if v)
    path.write_text("# Written by `swing init`. Keys live here; keep it private.\n"
                    + body + "\n")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def check_database(url: str) -> tuple[bool, str]:
    """Connect and confirm pgvector is available, without raising."""
    try:
        import psycopg
    except ImportError:                                   # pragma: no cover
        return False, "psycopg is not installed"
    try:
        with psycopg.connect(url, connect_timeout=5) as conn:
            row = conn.execute(
                "SELECT count(*) FROM pg_available_extensions WHERE name='vector'"
            ).fetchone()
            if not row or not row[0]:
                return False, ("connected, but the `vector` extension is not "
                               "available — use the pgvector/pgvector image")
            return True, "connected, pgvector available"
    except Exception as exc:   # noqa: BLE001 — any failure is a failed check
        return False, str(exc).strip().splitlines()[0]


def _ask(label: str, blurb: str, default: str, required: bool, current: str) -> str:
    """One prompt. Enter keeps the current value, then the default."""
    print(f"\n\033[1m{label}\033[0m")
    for line in blurb.splitlines():
        print(f"  {line}")
    shown = current or default
    suffix = f" [{shown}]" if shown else (" (required)" if required else " (optional)")
    while True:
        try:
            answer = input(f"  > {label}{suffix}: ").strip() or shown
        except EOFError:
            answer = shown
            print()
        if label == "SEC_USER_AGENT" and answer and "@" not in answer:
            print("  needs an email address in it — the SEC returns 403 otherwise.")
            continue
        if answer or not required:
            return answer
        print("  this one is required.")


def run(home: Path | None = None, *, force: bool = False,
        interactive: bool = True) -> int:
    from swing.paths import DEFAULT_HOME

    home = (home or DEFAULT_HOME).expanduser().resolve()
    print(f"Setting up swing in \033[1m{home}\033[0m")

    (home / "data").mkdir(parents=True, exist_ok=True)
    written = seed_config(home, force=force)
    print(f"\n  config  {len(written)} file(s) written"
          if written else "\n  config  already present, left alone")
    print("  data    ready")

    env_path = home / ".env"
    values = read_env(env_path)
    if interactive:
        for key, blurb, default, required in PROMPTS:
            values[key] = _ask(key, blurb, default, required, values.get(key, ""))
    else:
        for key, _, default, _required in PROMPTS:
            values.setdefault(key, default)

    write_env(env_path, values)
    print(f"\n  env     {env_path} (permissions 0600)")

    # ⚠️ Check this BEFORE the database. Every command loads Settings, and
    # Settings refuses an SEC contact with no email — so a missing one is not a
    # collector problem for later, it stops `swing` from starting at all. On a
    # clean install it surfaced as "1 validation error for Settings" from the
    # schema step and was reported as a bug.
    if "@" not in values.get("SEC_USER_AGENT", ""):
        print("  sec     ✗ SEC_USER_AGENT needs an email address in it")
        print(f"\n\033[1mNext:\033[0m add one to {env_path}, e.g.\n"
              "  SEC_USER_AGENT=Jane Smith jane@example.com\n"
              "then re-run `swing init`. The SEC blocks requests without it.\n")
        return 1
    print("  sec     ✓ contact set")

    url = values.get("DATABASE_URL", "")
    db_ok, detail = check_database(url) if url else (False, "no DATABASE_URL set")
    print(f"  db      {'✓' if db_ok else '✗'} {detail}")

    schema_error = ""
    if db_ok:
        os_environ_apply(values)
        try:
            from swing.store.session import apply_schema

            apply_schema()
            print("  schema  applied")
        except Exception as exc:   # noqa: BLE001 — report, never traceback
            schema_error = str(exc).strip().splitlines()[0]
            print(f"  schema  ✗ {schema_error}")

    print(_next_steps(home, db_ok, schema_error,
                      bool(values.get("GEMINI_API_KEY"))))
    return 0 if db_ok and not schema_error else 1


def os_environ_apply(values: dict[str, str]) -> None:
    """Make the just-written settings visible to this process.

    `swing.paths` and `Settings` both cache, and `apply_schema` runs in the same
    process that wrote the .env, so without this the schema step reads whatever
    the environment held before init started.
    """
    import os

    for k, v in values.items():
        if v:
            os.environ.setdefault(k, v)
    from swing.common.settings import get_settings

    get_settings.cache_clear()


def _next_steps(home: Path, db_ok: bool, schema_error: str, has_key: bool) -> str:
    # ⚠️ These two failures are not the same and must not print the same advice.
    # A clean 0.1.0 install connected to Postgres fine and then failed on a
    # schema.sql that was never packaged — and was told to go start a database
    # it was already talking to.
    if db_ok and schema_error:
        return (
            "\n\033[1mNext:\033[0m the database is fine, but the schema could not "
            "be applied:\n"
            f"  {schema_error}\n"
            "  This is a bug — please report it at\n"
            "  https://github.com/spoigai21/swing-agent/issues\n"
        )
    if not db_ok:
        return (
            "\n\033[1mNext:\033[0m the database is not reachable yet.\n"
            "  1. Start one:  docker compose up -d\n"
            "     (or point DATABASE_URL at your own Postgres — it needs the\n"
            "      pgvector extension, so use the pgvector/pgvector:pg16 image)\n"
            "  2. Re-run:     swing init\n"
        )
    lines = [
        "\n\033[1mNext:\033[0m",
        "  1. swing backfill                 # price history",
        "  2. swing backfill-events          # SEC filings + world news, years back",
        "  3. swing collect --daemon &       # keep it running; news cannot be",
        "                                    # backfilled once it has scrolled away",
        "  4. swing why NVDA                 # ask it something",
    ]
    if not has_key:
        lines.append("\n  No GEMINI_API_KEY set, so `why` and `ask` will not write an")
        lines.append(f"  explanation yet. Add it to {home / '.env'} when you have one.")
    return "\n".join(lines) + "\n"
