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

import os
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
     ("Google Gemini API key — the only AI provider swing supports. Free at\n"
      "https://aistudio.google.com/apikey (free tier: 20 requests/day).\n"
      "It is checked as soon as you paste it, without using any of those 20.\n"
      "Skip it for now and add it later with `swing key`."),
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


GEMINI_KEY_URL = "https://aistudio.google.com/apikey"

#: Values that must never be echoed back to the terminal in full.
SECRETS = {"GEMINI_API_KEY", "FINNHUB_API_KEY"}


def mask(value: str) -> str:
    return f"…{value[-4:]}" if len(value) > 8 else "set"


def default_model() -> str:
    from swing.common.settings import Settings

    return Settings.model_fields["attribution_model"].default


def check_gemini_key(key: str, model: str | None = None) -> tuple[bool, str]:
    """Is this key accepted, and can it reach the model? Costs no quota.

    `models.get` is a metadata call, not generation, so it does not touch the
    free tier's 20 requests a day — a user can check a key as often as they
    like. It also catches the second failure mode: a valid key on a project
    that cannot see the configured model.
    """
    model = model or default_model()
    try:
        from google import genai
    except ImportError:                                   # pragma: no cover
        return False, "the google-genai package is not installed"
    # ⚠️ Keep the client in a variable. `genai.Client(...).models.get(...)`
    # lets the client be collected mid-call and fails with "the client has been
    # closed", which reads exactly like a bad key.
    client = genai.Client(api_key=key)
    try:
        client.models.get(model=model)
    except Exception as exc:   # noqa: BLE001 — every failure becomes a sentence
        text = str(exc)
        if "API key not valid" in text or "API_KEY_INVALID" in text:
            return False, "Google rejected this key — check you copied all of it"
        if getattr(exc, "code", None) == 404:
            return False, f"the key works, but it cannot see {model}"
        if getattr(exc, "code", None) == 403:
            return False, ("the key was refused (403) — the Generative Language "
                           "API may be disabled on its project")
        return False, text.strip().splitlines()[0][:160]
    return True, f"key works, {model} is available"


def set_env_value(path: Path, key: str, value: str) -> None:
    """Set one variable, leaving every other line — comments included — alone."""
    lines = path.read_text().splitlines() if path.is_file() else []
    out, done = [], False
    for line in lines:
        name = line.split("=", 1)[0].strip()
        if not line.lstrip().startswith("#") and "=" in line and name == key:
            if not done:
                out.append(f"{key}={value}")
                done = True
            continue
        out.append(line)
    if not done:
        out.append(f"{key}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _validate(label: str, answer: str) -> str | None:
    """None when fine, else the reason to ask again."""
    if label == "SEC_USER_AGENT" and "@" not in answer:
        return "needs an email address in it — the SEC returns 403 otherwise."
    if label == "GEMINI_API_KEY":
        ok, detail = check_gemini_key(answer)
        print(f"    {'✓' if ok else '✗'} {detail}")
        if not ok:
            return "paste it again, or press Enter with nothing to skip for now."
    return None


def _ask(label: str, blurb: str, default: str, required: bool, current: str) -> str:
    """One prompt. Enter keeps the current value, then the default."""
    import getpass

    secret = label in SECRETS
    print(f"\n\033[1m{label}\033[0m")
    for line in blurb.splitlines():
        print(f"  {line}")
    shown = current or default
    if shown:
        suffix = f" [{mask(shown) if secret else shown}]"
    else:
        suffix = " (required)" if required else " (optional, Enter to skip)"
    if secret:
        suffix += " — typing is hidden, paste and press Enter"
    while True:
        try:
            raw = (getpass.getpass if secret else input)(f"  > {label}{suffix}: ")
        except EOFError:
            raw = ""
            print()
        answer = raw.strip().strip("'\"") or shown
        if not answer:
            if not required:
                return ""
            print("  this one is required.")
            continue
        if answer != current and (why := _validate(label, answer)):
            print(f"  {why}")
            if label == "GEMINI_API_KEY" and not raw.strip():
                return ""
            continue
        return answer


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
        # ⚠️ The ENVIRONMENT comes first in non-interactive mode. Reading only
        # the .env file made `swing init --non-interactive` unusable from CI, a
        # Dockerfile or any provisioning script: the values were exported and
        # ignored, and setup failed asking for a contact it had been given.
        for key, _, default, _required in PROMPTS:
            if not values.get(key):
                values[key] = os.environ.get(key, "") or default

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
        lines.append("\n  No Gemini key yet, so `why` and `ask` cannot write an explanation.")
        lines.append(f"  Get one free at {GEMINI_KEY_URL}, then run:  swing key")
    return "\n".join(lines) + "\n"


def key_command(value: str | None = None, *, check: bool = False,
                verify: bool = True) -> int:
    """`swing key` — add, replace or test the Gemini API key.

    One command instead of "find the hidden .env and edit it". The key is
    checked against Google before it is saved (free: no generation happens),
    so a typo is caught here rather than as a vague failure on the next `why`.
    """
    import getpass
    import os

    from swing.paths import ENV_FILE

    saved = read_env(ENV_FILE).get("GEMINI_API_KEY", "").strip("'\"")
    model = read_env(ENV_FILE).get("ATTRIBUTION_MODEL", "").strip("'\"") or default_model()
    exported = os.environ.get("GEMINI_API_KEY", "")

    if check:
        key = exported or saved
        if not key:
            print(f"No Gemini API key set. Get one free at {GEMINI_KEY_URL}, "
                  "then run:  swing key")
            return 1
        where = "your shell environment" if exported else str(ENV_FILE)
        ok, detail = check_gemini_key(key, model)
        print(f"{'✓' if ok else '✗'} {detail}  (key {mask(key)}, from {where})")
        return 0 if ok else 1

    if not value:
        print("Gemini is the only AI provider swing supports.")
        print(f"Get a free key at {GEMINI_KEY_URL} — it starts with 'AIza'.")
        try:
            value = getpass.getpass("Paste it here (typing is hidden), then Enter: ")
        except EOFError:
            value = ""
    value = (value or "").strip().strip("'\"")
    if not value:
        print("Nothing entered; key unchanged.")
        return 1

    if verify:
        ok, detail = check_gemini_key(value, model)
        print(f"{'✓' if ok else '✗'} {detail}")
        if not ok:
            print("Not saved. Run `swing key` again with the corrected key, or add "
                  "--no-verify to save it anyway.")
            return 1

    set_env_value(ENV_FILE, "GEMINI_API_KEY", value)
    print(f"Saved to {ENV_FILE} (key {mask(value)}, readable only by you).")

    # ⚠️ An exported variable beats the .env file, so a stale key in the shell
    # would silently keep winning and the new one would appear not to work.
    if exported and exported != value:
        print("\n⚠ GEMINI_API_KEY is also set in your shell, and that copy wins over "
              "the saved one.\n  Remove it (e.g. `unset GEMINI_API_KEY`, and delete "
              "the export from your shell profile) so the new key is used.")
    return 0
