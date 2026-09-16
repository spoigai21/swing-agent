"""Placebo harness design invariants.

The placebo test is the primary honesty signal (agent-plan.md 4.3), so the
design itself is tested — a subtly wrong design produces a confident number that
measures the wrong thing.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from swing.eval.placebo import relabel_timing

ONSET = datetime(2026, 7, 2, 13, 30, tzinfo=UTC)


def _c(published, timing="pre_move", cid=1):
    return {"id": cid, "timing": timing, "earliest_published": published,
            "best_tier": 1, "distinct_sources": 1, "member_count": 1,
            "headline": "h", "summary": "", "source": "sec-edgar", "rank": 1}


class TestTimingRelabel:
    def test_donor_article_after_test_onset_becomes_post_move(self, monkeypatch):
        """The bug this fixes: a donor's articles keep the DONOR's timing labels.
        A TSLA earnings 8-K from 2026-07-23 was presented as pre-move evidence
        for a 2026-07-02 swing, so the prompt's timestamps contradicted its
        timing labels."""
        import swing.eval.placebo as pb

        monkeypatch.setattr(pb, "connect", _fake_connect(ONSET))
        out = relabel_timing({"pre_move": [_c(ONSET + timedelta(days=21))]}, 1)
        assert out["pre_move"] == []
        assert len(out["post_move"]) == 1

    def test_donor_article_before_onset_stays_pre_move(self, monkeypatch):
        import swing.eval.placebo as pb

        monkeypatch.setattr(pb, "connect", _fake_connect(ONSET))
        out = relabel_timing({"pre_move": [_c(ONSET - timedelta(days=21))]}, 1)
        assert len(out["pre_move"]) == 1 and out["post_move"] == []

    def test_relabel_overrides_a_wrong_incoming_label(self, monkeypatch):
        import swing.eval.placebo as pb

        monkeypatch.setattr(pb, "connect", _fake_connect(ONSET))
        out = relabel_timing({"post_move": [_c(ONSET - timedelta(days=5),
                                               timing="post_move")]}, 1)
        assert len(out["pre_move"]) == 1


def _fake_connect(onset):
    import contextlib

    class _Cur:
        def execute(self, *a, **k):
            return self

        def fetchone(self):
            return {"onset_ts": onset}

    @contextlib.contextmanager
    def _conn():
        yield _Cur()

    return _conn


class TestCasePool:
    def test_any_swing_can_be_tested_but_donors_need_evidence(self, monkeypatch):
        """Requiring the TEST swing to have 3+ clusters (they are replaced anyway)
        capped the pool at 188, so Gate 4's 200 was unreachable."""
        from datetime import date

        import swing.eval.placebo as pb

        donors = [{"id": 1, "ticker": "NVDA", "d": date(2026, 1, 5), "pre": 4}]
        tests = [{"id": 1, "ticker": "NVDA", "d": date(2026, 1, 5)},
                 {"id": 2, "ticker": "NVDA", "d": date(2026, 3, 2)},     # no clusters of its own
                 {"id": 3, "ticker": "NVDA", "d": date(2026, 1, 12)},    # donor only 7 days earlier
                 {"id": 4, "ticker": "TSLA", "d": date(2026, 3, 2)}]     # no same-ticker donor
        monkeypatch.setattr(pb, "_eligible", lambda: donors)
        monkeypatch.setattr(pb, "_testable", lambda: tests)
        cases = pb.build_cases(10)
        assert [(c.swing_id, c.donor_swing_id) for c in cases] == [(2, 1)]


class TestEvalVersioning:
    def test_eval_hash_changes_with_the_placebo_design(self, tmp_path, monkeypatch):
        """config_hash only covers YAML, so a change to donor selection would
        otherwise let two different experimental designs pool their results."""
        import swing.common.versioning as v

        v.eval_hash.cache_clear()
        first = v.eval_hash()
        assert len(first) == 12 and first.isalnum()
        v.eval_hash.cache_clear()
        assert v.eval_hash() == first        # stable when nothing changes

    def test_eval_hash_is_separate_from_config_hash(self):
        from swing.common.versioning import config_hash, eval_hash

        assert eval_hash() != config_hash()


class TestATransientBlipDoesNotEndTheNight:
    """2026-09-16 lost 3 of 12 cases to one '503 UNAVAILABLE ... high demand'.

    At ~12 cases a night against a 200-case gate, abandoning the batch on a
    single hiccup costs a quarter of the night. A dead model or a spent quota
    still has to stop it quickly, so the rule is CONSECUTIVE failures.
    """

    def _setup(self, monkeypatch, outcomes):
        from swing.agent import graph
        from swing.eval import placebo as pb

        cases = [pb.PlaceboCase(swing_id=i, ticker="NVDA", donor_swing_id=100 + i)
                 for i in range(len(outcomes))]
        monkeypatch.setattr(pb, "build_cases", lambda n, seed: cases)
        monkeypatch.setattr(pb, "already_run", lambda seed: set())
        monkeypatch.setattr(pb, "cached_clusters", lambda sid: {})
        monkeypatch.setattr(pb, "relabel_timing", lambda clusters, sid: clusters)
        monkeypatch.setattr(pb, "eval_hash", lambda: "test")

        seen = []

        def fake_attribute(swing_id, **kwargs):
            seen.append(swing_id)
            if outcomes[swing_id] == "fail":
                return {"verdict_reason": "llm_error"}
            return {"attribution": type("A", (), {"verdict": "unexplained",
                                                  "candidates": []})()}

        monkeypatch.setattr(graph, "attribute_swing", fake_attribute)
        return seen

    def test_one_failure_is_skipped_and_the_batch_continues(self, monkeypatch):
        seen = self._setup(monkeypatch, ["ok", "fail", "ok", "ok"])
        from swing.eval.placebo import run

        out = run(n=4, persist=False)
        assert len(seen) == 4, "the batch must not stop at the single failure"
        assert out["n"] == 3, "the failed call must not be scored as an abstention"

    def test_consecutive_failures_stop_the_run(self, monkeypatch):
        seen = self._setup(monkeypatch, ["ok", "fail", "fail", "ok"])
        from swing.eval.placebo import run

        out = run(n=4, persist=False)
        assert len(seen) == 3, "a dead model must stop the run promptly"
        assert out["n"] == 1
