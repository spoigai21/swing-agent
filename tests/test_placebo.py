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
