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
