"""The 0.1.0 release was pip-installable and immediately dead.

`config/` lives at the repo root, outside the package, so it is absent from the
wheel and `swing.paths` raised "cannot locate the project root" on the first
command a new user typed. It looked fine in testing only because `uvx` resolved
`swing` to the author's local editable checkout instead of the wheel it had just
downloaded — the package under test was never the package that shipped.

These tests are about the install nobody in this repo is running.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from swing import paths
from swing.interface import setup


class TestTheShippedDefaultsMatchTheRepo:
    """Two copies of the config exist: config/ is what this repo runs on and
    src/swing/defaults/ is what users get. They must not drift."""

    def test_every_config_file_ships(self):
        for name in ("watchlist.yaml", "sources.yaml", "thresholds.yaml"):
            assert (paths.PACKAGED_DEFAULTS / name).is_file(), \
                f"{name} is missing from the wheel; a fresh install cannot start"

    def test_the_prompts_ship_too(self):
        shipped = sorted(p.name for p in (paths.PACKAGED_DEFAULTS / "prompts").glob("*.md"))
        assert shipped, "no prompt shipped — `why` would have nothing to render"

    def test_the_two_copies_are_identical(self):
        repo = paths.ROOT / "config"
        if not repo.is_dir():                      # installed, not a checkout
            pytest.skip("not running from a source tree")
        for src in paths.PACKAGED_DEFAULTS.rglob("*"):
            if src.is_dir():
                continue
            twin = repo / src.relative_to(paths.PACKAGED_DEFAULTS)
            assert twin.read_bytes() == src.read_bytes(), (
                f"{twin} and {src} have drifted. Re-sync with:\n"
                "  cp config/*.yaml src/swing/defaults/ && "
                "cp config/prompts/* src/swing/defaults/prompts/")


class TestPathResolution:
    def test_swing_home_wins_when_it_is_valid(self, tmp_path, monkeypatch):
        (tmp_path / "config").mkdir()
        monkeypatch.setenv("SWING_HOME", str(tmp_path))
        paths.project_root.cache_clear()
        assert paths.project_root() == tmp_path.resolve()
        paths.project_root.cache_clear()

    def test_a_swing_home_without_config_says_what_to_do(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SWING_HOME", str(tmp_path))
        paths.project_root.cache_clear()
        with pytest.raises(RuntimeError, match="swing init"):
            paths.project_root()
        paths.project_root.cache_clear()

    def test_an_installed_user_falls_back_to_a_home_that_does_not_exist_yet(
            self, monkeypatch):
        """⚠️ Must NOT raise. `swing init` has to run before ~/.swing exists,
        and raising here is exactly what made 0.1.0 unusable."""
        monkeypatch.delenv("SWING_HOME", raising=False)
        monkeypatch.setattr(paths, "_checkout_root", lambda: None)
        paths.project_root.cache_clear()
        assert paths.project_root() == paths.DEFAULT_HOME
        paths.project_root.cache_clear()


class TestInitProducesAWorkingHome:
    def _init(self, tmp_path):
        return setup.run(tmp_path / "home", interactive=False)

    def test_it_writes_the_config_a_fresh_install_needs(self, tmp_path):
        self._init(tmp_path)
        home = tmp_path / "home"
        for name in ("watchlist.yaml", "sources.yaml", "thresholds.yaml"):
            assert (home / "config" / name).is_file()
        assert (home / "data").is_dir()

    def test_the_env_file_is_not_world_readable(self, tmp_path):
        """It holds API keys and a database password."""
        self._init(tmp_path)
        mode = (tmp_path / "home" / ".env").stat().st_mode
        assert not mode & 0o077, f"mode {oct(mode)} exposes secrets to other users"

    def test_re_running_never_clobbers_your_edits(self, tmp_path):
        self._init(tmp_path)
        watchlist = tmp_path / "home" / "config" / "watchlist.yaml"
        watchlist.write_text("# my own watchlist\n")
        self._init(tmp_path)
        assert watchlist.read_text() == "# my own watchlist\n"

    def test_force_restores_the_shipped_defaults(self, tmp_path):
        self._init(tmp_path)
        watchlist = tmp_path / "home" / "config" / "watchlist.yaml"
        watchlist.write_text("# broken\n")
        setup.run(tmp_path / "home", interactive=False, force=True)
        assert watchlist.read_bytes() == (paths.PACKAGED_DEFAULTS / "watchlist.yaml").read_bytes()

    def test_an_existing_key_survives_a_re_run(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        setup.write_env(home / ".env", {"GEMINI_API_KEY": "sk-keep-me"})
        setup.run(home, interactive=False)
        assert setup.read_env(home / ".env")["GEMINI_API_KEY"] == "sk-keep-me"

    def test_a_dead_database_reports_instead_of_raising(self, tmp_path):
        ok, detail = setup.check_database("postgresql://no:no@127.0.0.1:1/none")
        assert ok is False
        assert detail, "a failed check must say why"


class TestEveryCommandButInitRequiresSetup:
    def test_init_is_the_only_exemption(self):
        from swing import cli

        assert cli.NEEDS_NO_CONFIG == {"init"}

    def test_an_unconfigured_home_names_the_fix(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "WATCHLIST", tmp_path / "nope.yaml")
        with pytest.raises(SystemExit, match="swing init"):
            paths.require_initialised()


class TestEvalHashSurvivesBeingInstalled:
    def test_it_resolves_from_the_package_not_a_src_directory(self):
        """A wheel has no src/. The old form hashed an empty file for every
        installed user, pooling their results under one meaningless hash."""
        import inspect

        from swing.common import versioning

        src = inspect.getsource(versioning.eval_hash)
        assert 'ROOT / "src"' not in src
        assert versioning.eval_hash(), "must produce a hash, not an empty string"

    def test_the_hashed_file_actually_exists(self):
        import swing
        from swing.common.versioning import HASHED_EVAL_CODE

        package = Path(swing.__file__).resolve().parent
        for rel in HASHED_EVAL_CODE:
            assert (package / rel).is_file(), f"{rel} would hash as empty"


class TestBackfillFailsSoftly:
    """A backfill is the first thing a new user runs. It must degrade with an
    explanation, never a traceback."""

    def test_no_bigquery_project_skips_gdelt_and_says_why(self, monkeypatch):
        from swing.common import settings
        from swing.ingest import backfill_events

        monkeypatch.setattr(backfill_events, "edgar_history", lambda t: 7)
        monkeypatch.setattr("swing.ingest.normalize.normalize_all", lambda: {"written": 0})
        cfg = settings.get_settings().model_copy(update={"google_cloud_project": ""})
        monkeypatch.setattr(settings, "get_settings", lambda: cfg)

        res = backfill_events.run(months=12)
        assert res.filings == 7
        assert res.articles == 0
        assert any("GOOGLE_CLOUD_PROJECT" in n for n in res.notes)

    def test_the_budget_guard_stops_the_walk_rather_than_overspending(self, monkeypatch):
        from swing.ingest import backfill_events, gdelt

        monkeypatch.setattr(gdelt, "affordable", lambda: False)
        monkeypatch.setattr(gdelt, "spent_gb", lambda: 700.0)
        articles, done, asked, _gb, notes = backfill_events.gdelt_history(12, ["NVDA"])
        assert (articles, done, asked) == (0, 0, 12)
        assert any("budget" in n for n in notes)

    def test_a_single_bad_month_does_not_end_the_walk(self, monkeypatch):
        from swing.ingest import backfill_events, gdelt

        calls = []

        def flaky(symbols, start, end, **kw):
            calls.append(start)
            if len(calls) == 2:
                raise RuntimeError("bigquery hiccup")
            return 5

        monkeypatch.setattr(gdelt, "affordable", lambda: True)
        monkeypatch.setattr(gdelt, "spent_gb", lambda: 0.0)
        monkeypatch.setattr(gdelt, "fetch_window", flaky)
        articles, done, _asked, _gb, notes = backfill_events.gdelt_history(4, ["NVDA"])
        assert len(calls) == 4, "the walk must continue past a failed month"
        assert articles == 15 and done == 3
        assert any("hiccup" in n for n in notes)


class TestTheBackfillUsesSourcesThatHaveAPast:
    def test_edgar_asks_for_years_not_a_fortnight(self):
        """first_run_limit=40 stopped a fresh install at 40 filings. NVDA's
        submissions feed carries 62 8-Ks going back to 2020."""
        from swing.ingest.backfill_events import EDGAR_BACKFILL_LIMIT

        assert EDGAR_BACKFILL_LIMIT >= 1000

    def test_it_normalizes_so_the_rows_are_retrievable(self):
        import inspect

        from swing.ingest import backfill_events

        assert "normalize_all" in inspect.getsource(backfill_events.run), \
            "rows left in articles_raw are invisible to retrieval"


class TestTheDefaultModelIsTheOneThatWasMeasured:
    """The README's claims are per model. A default install must run the model
    those numbers came from, or it is advertising results it does not deliver."""

    def test_the_default_matches_what_the_metrics_were_measured_on(self):
        from swing.common.settings import Settings

        default = Settings.model_fields["attribution_model"].default
        assert default == "gemini-3.6-flash", (
            f"default is {default!r}; the README's confabulation and coverage "
            "numbers were measured on gemini-3.6-flash. Change both together.")


class TestAddingAGeminiKeyIsOneCommand:
    """`swing key` replaces "find the hidden .env and edit it by hand"."""

    def _home(self, tmp_path, monkeypatch, env_text=""):
        env = tmp_path / ".env"
        env.write_text(env_text)
        monkeypatch.setattr("swing.paths.ENV_FILE", env)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        return env

    def test_a_good_key_is_saved(self, tmp_path, monkeypatch, capsys):
        env = self._home(tmp_path, monkeypatch, "SEC_USER_AGENT=a a@b.c\n")
        monkeypatch.setattr(setup, "check_gemini_key", lambda k, m=None: (True, "ok"))
        assert setup.key_command("AIzaGOODKEY1234") == 0
        assert setup.read_env(env)["GEMINI_API_KEY"] == "AIzaGOODKEY1234"
        assert "AIzaGOODKEY1234" not in capsys.readouterr().out, "never echo the key"

    def test_a_bad_key_is_not_saved(self, tmp_path, monkeypatch):
        """Caught here, not as a vague failure on the next `why`."""
        env = self._home(tmp_path, monkeypatch, "GEMINI_API_KEY=AIzaOLDKEY9999\n")
        monkeypatch.setattr(setup, "check_gemini_key",
                            lambda k, m=None: (False, "Google rejected this key"))
        assert setup.key_command("AIzaTYPO") == 1
        assert setup.read_env(env)["GEMINI_API_KEY"] == "AIzaOLDKEY9999"

    def test_pasted_quotes_and_spaces_are_stripped(self, tmp_path, monkeypatch):
        env = self._home(tmp_path, monkeypatch)
        monkeypatch.setattr(setup, "check_gemini_key", lambda k, m=None: (True, "ok"))
        setup.key_command('  "AIzaQUOTED5678"  ')
        assert setup.read_env(env)["GEMINI_API_KEY"] == "AIzaQUOTED5678"

    def test_no_argument_prompts_with_hidden_input(self, tmp_path, monkeypatch):
        env = self._home(tmp_path, monkeypatch)
        monkeypatch.setattr(setup, "check_gemini_key", lambda k, m=None: (True, "ok"))
        monkeypatch.setattr("getpass.getpass", lambda prompt="": "AIzaHIDDEN0000")
        monkeypatch.setattr("builtins.input",
                            lambda *a: pytest.fail("a key must not be typed visibly"))
        assert setup.key_command(None) == 0
        assert setup.read_env(env)["GEMINI_API_KEY"] == "AIzaHIDDEN0000"

    def test_saving_leaves_the_rest_of_the_file_alone(self, tmp_path, monkeypatch):
        env = self._home(tmp_path, monkeypatch,
                         "# my notes\nDATABASE_URL=postgresql://x\n\nGEMINI_API_KEY=old\n")
        monkeypatch.setattr(setup, "check_gemini_key", lambda k, m=None: (True, "ok"))
        setup.key_command("AIzaNEWKEY4321")
        text = env.read_text()
        assert "# my notes" in text and "DATABASE_URL=postgresql://x" in text
        assert text.count("GEMINI_API_KEY=") == 1
        assert not env.stat().st_mode & 0o077

    def test_a_shell_export_that_would_override_it_is_called_out(
            self, tmp_path, monkeypatch, capsys):
        """An exported variable beats .env, so the new key would silently lose."""
        self._home(tmp_path, monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSTALE0000")
        monkeypatch.setattr(setup, "check_gemini_key", lambda k, m=None: (True, "ok"))
        setup.key_command("AIzaFRESH1111")
        assert "wins over the saved one" in capsys.readouterr().out

    def test_check_reports_without_saving(self, tmp_path, monkeypatch, capsys):
        env = self._home(tmp_path, monkeypatch, "GEMINI_API_KEY=AIzaCURRENT777\n")
        monkeypatch.setattr(setup, "check_gemini_key", lambda k, m=None: (True, "key works"))
        assert setup.key_command(check=True) == 0
        assert "…T777" in capsys.readouterr().out
        assert env.read_text() == "GEMINI_API_KEY=AIzaCURRENT777\n"


class TestInitNeverShowsAKey:
    def test_a_saved_key_is_masked_in_the_prompt(self, monkeypatch):
        prompts = []
        monkeypatch.setattr("getpass.getpass",
                            lambda prompt="": prompts.append(prompt) or "")
        setup._ask("GEMINI_API_KEY", "blurb", "", False, "AIzaSECRETVALUE9876")
        assert "SECRETVALUE" not in prompts[0]
        assert "…9876" in prompts[0]

    def test_masking_reveals_at_most_four_characters(self):
        assert setup.mask("AIzaSyABCDEFGHIJ1234") == "…1234"
        assert setup.mask("short") == "set"


class TestTheFailureMessageMatchesTheFailure:
    """Every failure said "Gemini may be overloaded, ask again in a few minutes"
    — wrong advice for a bad key, where waiting fixes nothing."""

    def test_each_kind_is_recognised(self):
        from swing.agent.llm import failure_kind

        assert failure_kind(RuntimeError("400 API key not valid.")) == "invalid_key"
        assert failure_kind(RuntimeError(
            "429 GenerateRequestsPerDayPerProjectPerModel-FreeTier")) == "daily_cap"
        assert failure_kind(RuntimeError("503 UNAVAILABLE")) == "unavailable"

    @pytest.mark.parametrize("kind, phrase", [
        ("invalid_key", "swing key"),
        ("daily_cap", "midnight Pacific"),
        ("unavailable", "overloaded"),
    ])
    def test_the_user_is_told_what_to_do(self, monkeypatch, kind, phrase):
        from swing.agent import llm
        from swing.interface import explain

        monkeypatch.setattr(llm, "_last_failure", kind)
        assert phrase in explain.model_unavailable([])

    def test_a_missing_key_names_the_command_that_fixes_it(self, monkeypatch):
        from swing.agent import llm
        from swing.common import settings

        cfg = settings.get_settings().model_copy(update={"gemini_api_key": ""})
        monkeypatch.setattr(llm, "get_settings", lambda: cfg)
        with pytest.raises(RuntimeError, match="swing key"):
            llm.get_llm()


class TestTheDefaultInstallRunsEveryUserCommand:
    """The README said `pip install swing-agent`; that install could not run
    `why` (numpy) or `backfill` (pandas), which lived behind extras."""

    def test_nothing_a_user_runs_is_hidden_behind_an_extra(self):
        import tomllib

        project = tomllib.loads((paths.ROOT / "pyproject.toml").read_text())["project"]
        base = " ".join(project["dependencies"])
        for needed in ("numpy", "pandas", "yfinance", "google-cloud-bigquery",
                       "langgraph", "langchain-google-genai",
                       "sentence-transformers", "datasketch"):
            assert needed in base, f"{needed} is imported by a user command"
