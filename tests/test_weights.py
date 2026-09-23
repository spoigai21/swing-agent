"""The ranking weights were hand-set and never fitted.

A 55-move answer key and a walk-forward split helper sat unused beside them.
Fitting is arithmetic over component scores the database already stores, so it
costs no quota — the only real risk is fitting the noise and believing it.
"""
from __future__ import annotations

from datetime import date

from swing.models import weights as W


def cl(sem=0.5, tim=0.0, tier=3, rel=False, true=False):
    return W.Cluster(semantic=sem, timing=tim, tier=tier, novelty=0.0,
                     related_only=rel, is_true=true)


def case(sid, d, clusters):
    return W.Case(swing_id=sid, d=d, clusters=clusters)


BASE = {"w_semantic": 1.0, "w_timing": 1.0, "w_tier": 0.5,
        "w_novelty": 0.0, "w_related_penalty": 0.5}


class TestScoring:
    def test_the_formula_matches_the_one_in_rank_py(self):
        c = cl(sem=0.8, tim=0.5, tier=2)
        assert c.score(BASE) == 1.0 * 0.8 + 1.0 * 0.5 + 0.5 * (5 - 2)

    def test_a_related_only_cluster_pays_the_penalty(self):
        own, rel = cl(sem=0.8), cl(sem=0.8, rel=True)
        assert own.score(BASE) - rel.score(BASE) == BASE["w_related_penalty"]

    def test_recall_counts_a_case_once_however_many_true_clusters(self):
        c = case(1, date(2026, 1, 1), [cl(sem=0.9, true=True), cl(sem=0.8, true=True)])
        assert W.recall_at_k([c], BASE, k=10) == 1.0

    def test_recall_at_1_is_strict(self):
        c = case(1, date(2026, 1, 1), [cl(sem=0.9), cl(sem=0.1, true=True)])
        assert W.recall_at_1 if False else W.recall_at_k([c], BASE, k=1) == 0.0
        assert W.recall_at_k([c], BASE, k=2) == 1.0


class TestTheSearchCannotCheat:
    def test_a_case_with_no_labelled_cluster_is_dropped(self):
        """It rewards every weight vector equally and inflates the denominator."""
        assert case(1, date(2026, 1, 1), [cl(), cl()]).scorable is False
        assert case(2, date(2026, 1, 1), [cl(true=True)]).scorable is True

    def test_ties_break_toward_the_weights_already_in_use(self, monkeypatch):
        """Otherwise a tie reports a change that buys nothing."""
        monkeypatch.setattr(W, "current_weights", lambda: dict(BASE))
        cases = [case(1, date(2026, 1, 1), [cl(sem=0.9, true=True), cl(sem=0.1)])]
        best, score = W.grid_search(cases, k=1)
        assert score == 1.0
        assert best == BASE

    def test_fit_refuses_a_sample_too_small_to_split(self, monkeypatch):
        monkeypatch.setattr(W, "load_cases",
                            lambda: [case(i, date(2026, 1, 1), [cl(true=True)])
                                     for i in range(3)])
        assert "too few" in W.fit()["error"]

    def test_training_swings_all_precede_the_held_out_ones(self, monkeypatch):
        cases = [case(i, date(2026, 1, 1 + i), [cl(sem=i / 20, true=i % 2 == 0), cl()])
                 for i in range(20)]
        monkeypatch.setattr(W, "load_cases", lambda: list(cases))
        monkeypatch.setattr(W, "current_weights", lambda: dict(BASE))
        r = W.fit()
        assert r["n_train"] + r["n_test"] == r["n_cases"]
        assert r["cutoff"] == str(date(2026, 1, 1 + r["n_train"]))


class TestTheVerdictResistsNoise:
    """§16.25: three versions of a Phase 5 verdict rule each blessed a win
    finer than the instrument could resolve."""

    def _r(self, cur, fit_, n_test=12):
        return {"test_current": cur, "test_fitted": fit_, "n_test": n_test,
                "changed": {"w_timing": (1.0, 0.5)}}

    def test_no_gain_keeps_the_current_weights(self):
        assert "do NOT beat" in W.verdict(self._r(0.83, 0.83))

    def test_a_loss_keeps_the_current_weights(self):
        assert "do NOT beat" in W.verdict(self._r(0.83, 0.75))

    def test_a_gain_under_one_test_case_is_inside_the_resolution(self):
        assert "resolution" in W.verdict(self._r(0.830, 0.860))

    def test_a_one_case_gain_is_called_suggestive_not_decisive(self):
        assert "not decisive" in W.verdict(self._r(0.250, 0.340))

    def test_a_clear_gain_names_the_search_width(self):
        out = W.verdict(self._r(0.250, 0.417))
        assert "Worth applying" in out and str(W.combos()) in out

    def test_an_unchanged_grid_says_the_hand_set_weights_won(self):
        r = self._r(0.9, 0.9)
        r["changed"] = {}
        assert "already win the grid" in W.verdict(r)


class TestItNeverWritesConfig:
    def test_nothing_here_edits_thresholds(self):
        """Applying changes config_hash and resets the Gate 4 placebo count, so
        it must be a deliberate human act, never a side effect of measuring."""
        import inspect

        src = inspect.getsource(W)
        for forbidden in ("write_text", "safe_dump", "yaml.dump"):
            assert forbidden not in src


class TestRecencyIsNotTheCatalyst:
    """`w_timing` was 1.0, and timing_score decays with a 12h half-life, so the
    most RECENT pre-move article almost always led the ranking. Measured over 43
    labelled swings (13 held out) it was a clean dose-response on both splits:

        w_timing   train@1  held@1
             0.0     0.655   0.615
             0.1     0.552   0.538
             1.0     0.241   0.308   <- the old value
             1.5     0.138   0.231

    Production recall@1 went 0.308 -> 0.548 on the rebuild.
    """

    def test_timing_is_a_tiebreak_not_the_ranking(self):
        from swing.ingest.config import thresholds

        w = float(thresholds()["ranking"]["w_timing"])
        assert 0.0 < w <= 0.25, (
            f"w_timing={w}: above ~0.25 the newest article wins by default, and "
            "the true catalyst usually is not the newest one")

    def test_semantic_still_outweighs_it(self):
        from swing.ingest.config import thresholds

        cfg = thresholds()["ranking"]
        assert float(cfg["w_semantic"]) > float(cfg["w_timing"])

    def test_post_move_articles_still_earn_nothing_from_timing(self):
        """The weight changed; the rule that timing_score is 0 at or after onset
        did not. That rule is what enforces "published before the move"."""
        from datetime import UTC, datetime, timedelta

        from swing.analysis.dedup import ClusterView
        from swing.analysis.rank import timing_score

        onset = datetime(2026, 9, 18, 13, 30, tzinfo=UTC)
        after = ClusterView.__new__(ClusterView)
        after.earliest_published = onset + timedelta(hours=1)
        assert timing_score(after, onset) == 0.0
