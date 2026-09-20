"""Walk-forward splits. The failure this prevents is silent and flattering.

A shuffled split on a time series leaks tomorrow into training, so a broken
model scores brilliantly and ships. agent-plan.md 5.3 is explicit: "Never
shuffle." These tests pin that the module cannot be talked into it.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from swing.models.splits import Split, assert_time_forward, expanding_folds, split_at

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _rows(n: int, key: str = "published_at") -> list[dict]:
    return [{key: T0 + timedelta(days=i), "id": i} for i in range(n)]


class TestSplitAt:
    def test_cutoff_is_exclusive_for_train_inclusive_for_test(self):
        s = split_at(_rows(10), T0 + timedelta(days=6))
        assert [r["id"] for r in s.train] == [0, 1, 2, 3, 4, 5]
        assert [r["id"] for r in s.test] == [6, 7, 8, 9]

    def test_everything_before_the_data_gives_an_empty_train_side(self):
        s = split_at(_rows(5), T0 - timedelta(days=1))
        assert s.train == [] and len(s.test) == 5

    def test_a_row_missing_its_timestamp_is_an_error_not_a_guess(self):
        with pytest.raises(ValueError, match="time-forward"):
            split_at([{"published_at": None}], T0)


class TestLeakageIsAnError:
    def test_overlapping_sets_raise(self):
        rows = _rows(10)
        with pytest.raises(ValueError, match="leakage"):
            assert_time_forward(rows[5:], rows[:5], "published_at")

    def test_touching_sets_raise(self):
        """Equal timestamps on both sides still leak: the model sees a row from
        the same instant it is asked to predict."""
        same = [{"published_at": T0, "id": 1}]
        with pytest.raises(ValueError, match="leakage"):
            assert_time_forward(same, [{"published_at": T0, "id": 2}], "published_at")

    def test_an_empty_side_is_not_leakage(self):
        assert_time_forward([], _rows(3), "published_at")
        assert_time_forward(_rows(3), [], "published_at")


class TestExpandingFolds:
    def test_every_fold_is_time_forward(self):
        for fold in expanding_folds(_rows(40), n_folds=4):
            newest_train = max(r["published_at"] for r in fold.train)
            oldest_test = min(r["published_at"] for r in fold.test)
            assert newest_train < oldest_test

    def test_the_training_window_only_grows(self):
        folds = expanding_folds(_rows(40), n_folds=4)
        sizes = [len(f.train) for f in folds]
        assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)

    def test_rows_sharing_the_cutoff_instant_do_not_straddle_it(self):
        # Ten rows at the same timestamp in the middle: a naive index split
        # would put some on each side and leak.
        rows = (_rows(10) + [{"published_at": T0 + timedelta(days=10), "id": 100 + i}
                             for i in range(10)]
                + [{"published_at": T0 + timedelta(days=20 + i), "id": 200 + i} for i in range(10)])
        for fold in expanding_folds(rows, n_folds=3):
            assert max(r["published_at"] for r in fold.train) < min(
                r["published_at"] for r in fold.test)

    def test_too_few_rows_is_an_error(self):
        with pytest.raises(ValueError, match="at least"):
            expanding_folds(_rows(2), n_folds=4)


def test_no_split_function_accepts_a_shuffle_knob():
    """The guarantee is structural: no callable here takes shuffle or
    random_state, so a caller cannot opt into leakage even by accident.

    (Checking the source text instead would fail on the docstrings, which say
    "Never shuffle" — a test that trips over its own prose.)"""
    import inspect

    from swing.models import splits

    for name in splits.__all__:
        obj = getattr(splits, name)
        if not callable(obj):
            continue
        params = set(inspect.signature(obj).parameters)
        assert not params & {"shuffle", "random_state", "seed", "random"}, (
            f"{name} must not offer a way to shuffle a time series")


def test_split_repr_is_readable_in_training_logs():
    s = Split(train=[1, 2], test=[3], cutoff=T0)
    assert "train=2" in repr(s) and "test=1" in repr(s)


class TestEventLabelsCannotLeak:
    """Labels come FROM the 8-K Item numbers, and EDGAR headlines print them.

    Training on the raw headline lets the model read the answer off the input:
    0.900 accuracy leaky vs 0.835 stripped. The dataset therefore strips them,
    rather than relying on each training script to remember.
    """

    def test_item_numbers_never_survive_into_the_text(self):
        from swing.models.events import strip_label_leakage

        out = strip_label_leakage("NVDA 8-K — Item 2.02,9.01 — FORM 8-K Report date 2026-08-27")
        assert "2.02" not in out and "Item" not in out
        assert "8-K" not in out and "2026-08-27" not in out

    def test_real_prose_is_kept(self):
        from swing.models.events import strip_label_leakage

        out = strip_label_leakage(
            "CMG 8-K — Item 5.02 — CHIPOTLE APPOINTS SABIR SAMI TO ITS BOARD OF DIRECTORS")
        assert "APPOINTS SABIR SAMI" in out and "5.02" not in out

    def test_the_item_to_event_mapping(self):
        from swing.models.events import event_type_from_items

        assert event_type_from_items("2.02,9.01") == "earnings"
        assert event_type_from_items("5.02") == "management"
        assert event_type_from_items("2.01,9.01") == "m_and_a"
        assert event_type_from_items("8.01,9.01") == "other"
        # 9.01 only accompanies other items and is never itself the event.
        assert event_type_from_items("9.01") is None
        assert event_type_from_items(None) is None
        # An earnings release with a Reg FD courtesy copy is still earnings.
        assert event_type_from_items("2.02,7.01,9.01") == "earnings"


class TestSourceBasedLabels:
    """analyst-ratings rows are analyst actions by construction — a fifth class
    that 8-K Item numbers cannot express. Unlike the Item-number leak, the label
    comes from provenance while the text independently describes the event."""

    def test_the_source_map_covers_analyst_ratings(self):
        from swing.models.events import SOURCE_LABELS

        assert SOURCE_LABELS["analyst-ratings"] == "analyst_action"

    def test_item_labels_still_win_when_both_could_apply(self):
        """An 8-K is labelled by its Items even if its source were mapped: the
        filing's own Item numbers are the more specific evidence."""
        from swing.models.events import SOURCE_LABELS, event_type_from_items

        assert event_type_from_items("2.02,9.01") == "earnings"
        assert "sec-edgar" not in SOURCE_LABELS


class TestAMarginSmallerThanOneRowIsNotAResult:
    """The 5.2 verdict rule has been tightened twice, both times after it
    announced a pass it had not earned: first comparing one transformer seed to
    one lucky baseline run, then clearing the baseline by 0.010 when a single
    row of the smallest class was worth 0.0143."""

    def test_resolution_is_one_row_of_the_smallest_class(self):
        from swing.models.classifier import one_row_resolution

        support = {"analyst_action": 488, "earnings": 55, "m_and_a": 14,
                   "management": 43, "other": 52}
        assert one_row_resolution(support) == pytest.approx((1 / 5) / 14, rel=1e-6)

    def test_the_real_margin_that_looked_like_a_pass_is_rejected(self):
        from swing.models.classifier import verdict

        # distilbert min seed 0.846 vs pinned baseline 0.836.
        assert verdict(margin=0.010, resolution=0.0143,
                       mean_got=0.861, mean_base=0.836) == "below_resolution"

    def test_a_margin_above_one_row_passes(self):
        from swing.models.classifier import verdict

        assert verdict(margin=0.02, resolution=0.0143,
                       mean_got=0.870, mean_base=0.836) == "pass"

    def test_losing_on_every_seed_is_not_dressed_up(self):
        from swing.models.classifier import verdict

        assert verdict(-0.01, 0.0143, mean_got=0.84, mean_base=0.836) == "inside_noise"
        assert verdict(-0.05, 0.0143, mean_got=0.79, mean_base=0.836) == "loses"

    def test_empty_support_does_not_divide_by_zero(self):
        from swing.models.classifier import one_row_resolution

        assert one_row_resolution({}) == 0.0


class TestCatalystPairsAndSaturation:
    """5.3's stated baseline is the off-the-shelf embedder. On the 38 pairs that
    exist it already scores recall@10 = 1.000, so fine-tuning cannot be shown to
    beat it at k=10 however many pairs are collected. Detecting that is the
    point: it is a measurement problem, not a modelling one."""

    def _pairs(self, ranks):
        from swing.models.pairs import CatalystPair

        return [CatalystPair(swing_id=i, ticker="NVDA", positive_cluster_id=i,
                             positive_rank=r, negative_cluster_ids=[])
                for i, r in enumerate(ranks)]

    def test_recall_at_k_and_mrr(self):
        from swing.models.pairs import baseline_ranking

        m = baseline_ranking(self._pairs([1, 2, 3, 10]))
        assert m["recall@1"] == pytest.approx(0.25)
        assert m["recall@3"] == pytest.approx(0.75)
        assert m["recall@10"] == pytest.approx(1.0)
        assert m["MRR"] == pytest.approx((1 + 1 / 2 + 1 / 3 + 1 / 10) / 4)

    def test_a_perfect_baseline_is_reported_as_saturated(self):
        from swing.models.pairs import KS, baseline_ranking

        # Worst rank 7: saturated at k=10 only. recall@5 is 3/4 here, which is
        # exactly the real shape — the live pairs top out at rank 7 too.
        m = baseline_ranking(self._pairs([1, 2, 3, 7]))
        assert [k for k in KS if m[f"recall@{k}"] == 1.0] == [10]
        assert m["recall@5"] == pytest.approx(0.75)

    def test_several_k_saturate_together_when_every_rank_is_tight(self):
        from swing.models.pairs import KS, baseline_ranking

        m = baseline_ranking(self._pairs([1, 2, 3, 3]))
        assert [k for k in KS if m[f"recall@{k}"] == 1.0] == [3, 5, 10]
        assert 1 not in [k for k in KS if m[f"recall@{k}"] == 1.0], (
            "positives below rank 1 must leave headroom at k=1")

    def test_headroom_remains_at_rank_one(self):
        """Only 11 of 38 positives rank first, and the agent cites from the top
        of the list, so recall@1 is where fine-tuning could still pay."""
        from swing.models.pairs import baseline_ranking

        m = baseline_ranking(self._pairs([1] * 11 + [2] * 11 + [3] * 9 + [4, 4, 4, 5, 6, 6, 7]))
        assert m["recall@10"] == pytest.approx(1.0)
        assert m["recall@1"] == pytest.approx(11 / 38, abs=0.01)

    def test_empty_input_does_not_crash(self):
        from swing.models.pairs import baseline_ranking

        assert baseline_ranking([]) == {}


class TestContaminationRisk:
    """Step 4.4's first mitigation, as a diagnostic rather than a filter.

    It cannot observe what the model memorised — nothing can — so it ranks by
    press volume and lets a metric be read on the obscure tail as a check that
    recall is not doing the evidence's work.
    """

    def test_a_mega_cap_earnings_blowout_scores_high(self):
        from swing.eval.contamination import score

        r = score("AAPL", residual_z=5.3, earnings_mode=True)
        assert r.score > 0.8 and not r.low
        assert "mega-cap name" in r.reasons and "earnings day" in r.reasons

    def test_an_obscure_small_move_scores_low(self):
        from swing.eval.contamination import score

        r = score("SNDK", residual_z=2.1, earnings_mode=False)
        assert r.low, "a quiet move in a barely-covered name is unlikely to be recalled"

    def test_prominence_outweighs_magnitude(self):
        """An obscure ticker's big move still gets little ink, so it must not
        outrank a mega-cap's ordinary one."""
        from swing.eval.contamination import score

        assert score("U", 6.0).score < score("TSLA", 2.5).score

    def test_a_decimal_from_postgres_does_not_crash(self):
        """residual_z arrives as Decimal; it will not mix with float arithmetic."""
        from decimal import Decimal

        from swing.eval.contamination import score

        assert score("NVDA", Decimal("4.25")).score > 0

    def test_an_unknown_ticker_is_treated_as_obscure(self):
        from swing.eval.contamination import DEFAULT_PROMINENCE, score

        assert score("ZZZZ", 2.0).score <= DEFAULT_PROMINENCE

    def test_it_does_not_live_in_the_hashed_eval_file(self):
        """Editing eval/placebo.py discards every accumulated Gate 4 case."""
        from swing.common.versioning import HASHED_EVAL_CODE

        assert "eval/contamination.py" not in HASHED_EVAL_CODE


class TestAbstentionRunner:
    def test_it_targets_only_unattributed_no_catalyst_swings(self):
        import inspect

        from swing.eval import abstention

        src = inspect.getsource(abstention.pending)
        assert "n.no_catalyst" in src
        assert "run_kind = 'production'" in src
        assert "NOT EXISTS" in src, "already-attributed swings must not be re-spent"

    def test_it_stops_rather_than_burning_the_day_on_refusals(self):
        import inspect

        from swing.eval import abstention

        assert abstention.MAX_CONSECUTIVE_FAILURES >= 2
        assert "llm_error" in inspect.getsource(abstention.run)


class TestARetryCostsACase:
    """Under a hard 20/day cap, a retry is not free — it is a case not scored.

    On 2026-09-20 the day's last 10 requests produced 2 scored cases: three of
    them were retried twice each against 503 "high demand", spending 5 extra
    requests to rescue 1 case. Retrying the same swing is only worth it when
    someone is waiting for THAT swing.
    """

    def _quiet_limiter(self, monkeypatch, tmp_path):
        from swing.agent import llm

        monkeypatch.setattr(llm, "_llm_limiter", type("S", (), {"wait": lambda s: None})())
        monkeypatch.setattr(llm, "QUOTA_PATH", tmp_path / "usage.json")
        monkeypatch.setattr(llm, "_daily_cap_day", None, raising=False)

    def test_interactive_use_still_retries(self, monkeypatch, tmp_path):
        import pytest

        from swing.agent import llm

        self._quiet_limiter(monkeypatch, tmp_path)
        calls = []

        class Flaky:
            def invoke(self, prompt):
                calls.append(prompt)
                raise RuntimeError("503 UNAVAILABLE: model is overloaded")

        with pytest.raises(RuntimeError):
            llm.invoke_with_retry.retry_with(wait=lambda *a, **k: 0)(Flaky(), "p")
        assert len(calls) == 3, "a person at the prompt is worth three tries"

    def test_a_batch_spends_the_request_on_the_next_case_instead(
            self, monkeypatch, tmp_path):
        import pytest

        from swing.agent import llm

        self._quiet_limiter(monkeypatch, tmp_path)
        calls = []

        class Flaky:
            def invoke(self, prompt):
                calls.append(prompt)
                raise RuntimeError("503 UNAVAILABLE: model is overloaded")

        with llm.batch_mode(), pytest.raises(RuntimeError):
            llm.invoke_with_retry(Flaky(), "p")
        assert len(calls) == 1, "the other two requests belong to other cases"

    def test_the_ledger_believes_google_over_itself(self, monkeypatch, tmp_path):
        """The ledger counts this process's attempts, so it drifts. A daily-cap
        refusal is the one moment the true remaining count is known."""
        import json

        from swing.agent import llm

        self._quiet_limiter(monkeypatch, tmp_path)
        llm.record_call(3)
        assert llm.remaining_today() == llm.DAILY_QUOTA - 3
        try:
            llm.note_daily_cap()
            assert llm.remaining_today() == 0
            assert llm.daily_cap_reached()
            assert json.loads((tmp_path / "usage.json").read_text())[
                llm._today()] == llm.DAILY_QUOTA
        finally:
            llm._daily_cap_day = None


class TestTheDailyCapEndsARunButAServerBlipDoesNot:
    """Two failures look identical from the runner and mean opposite things:
    the cap means come back tomorrow, a 503 means try the next case."""

    def _runner(self, monkeypatch, verdicts, capped):
        from swing.agent import graph, llm
        from swing.eval import abstention

        asked = []

        def fake_attribute(swing_id, **kw):
            asked.append(swing_id)
            v = verdicts.pop(0)
            if v == "llm_error":
                return {"verdict_reason": "llm_error"}
            return {"attribution": type("A", (), {"verdict": v})()}

        monkeypatch.setattr(graph, "attribute_swing", fake_attribute)
        monkeypatch.setattr(llm, "daily_cap_reached", lambda: capped)
        monkeypatch.setattr(abstention, "pending", lambda: [1, 2, 3, 4, 5, 6])
        return abstention, asked

    def test_the_daily_cap_stops_at_the_first_failure(self, monkeypatch):
        abstention, asked = self._runner(
            monkeypatch, ["llm_error"] * 6, capped=True)
        result = abstention.run()
        assert asked == [1], "asking again cannot work until midnight PT"
        assert result["attributed"] == 0

    def test_a_server_blip_moves_on_to_the_next_case(self, monkeypatch):
        abstention, asked = self._runner(
            monkeypatch,
            ["llm_error", "unexplained", "llm_error", "unexplained",
             "unexplained", "unexplained"],
            capped=False)
        result = abstention.run()
        assert len(asked) == 6, "a blip on one swing says nothing about the next"
        assert result["attributed"] == 4


class TestEarningsFiguresMatchTheFiling:
    """Step 1.7's deltas are only worth having if they are the RIGHT quarter.

    Both failures below shipped in the first draft and were caught by checking
    the extraction against the company's own press-release text, which the
    corpus already holds:

      NVDA  release said $96.2 billion, extraction returned $3.10B from 2020
      SNDK  release said $8.97 billion, extraction returned April's $5.95B
    """

    def _facts(self, concept_rows):
        return {"facts": {"us-gaap": {c: {"units": {"USD": rows}}
                                      for c, rows in concept_rows.items()}}}

    def test_the_live_concept_wins_over_a_dead_one(self):
        """NVDA abandoned one tag in 2020 and AAPL abandoned the other in 2018,
        so "first concept with any rows" picks a dead series half the time."""
        from datetime import date

        from swing.analysis.earnings import _revenue

        facts = self._facts({
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                {"start": "2019-10-28", "end": "2020-01-26", "val": 3_100_000_000}],
            "Revenues": [
                {"start": "2026-04-27", "end": "2026-07-26", "val": 96_220_000_000}],
        })
        rows = _revenue(facts, date(2026, 8, 26))
        assert rows and rows[-1].value == 96_220_000_000, "must not serve the 2020 series"

    def test_a_quarter_too_old_for_the_filing_is_refused(self):
        """SNDK's fiscal Q4 is reported annually, so it is absent from quarterly
        data. Returning the PRIOR quarter as though it were the announced one is
        worse than returning nothing."""
        from datetime import date

        from swing.analysis.earnings import MAX_PERIOD_LAG_DAYS, _at_or_before, quarters

        facts = self._facts({"Revenues": [
            {"start": "2026-01-03", "end": "2026-04-03", "val": 5_950_000_000}]})
        rows = quarters(facts, "Revenues")
        latest = _at_or_before(rows, date(2026, 8, 5))
        assert latest is not None
        assert (date(2026, 8, 5) - latest.end).days > MAX_PERIOD_LAG_DAYS

    def test_annual_durations_never_count_as_a_quarter(self):
        from swing.analysis.earnings import quarters

        facts = self._facts({"Revenues": [
            {"start": "2025-08-29", "end": "2026-05-28", "val": 78_959_000_000},
            {"start": "2026-02-27", "end": "2026-05-28", "val": 41_456_000_000},
        ]})
        rows = quarters(facts, "Revenues")
        assert [r.value for r in rows] == [41_456_000_000], "9-month period must be dropped"

    def test_the_sentence_never_implies_a_beat_or_miss(self):
        """Consensus is paid, so the line must not read as surprise-vs-estimate."""
        from datetime import date

        from swing.analysis.earnings import Deltas

        s = Deltas(ticker="NVDA", period_end=date(2026, 7, 26), revenue=96_220_000_000,
                   revenue_qoq=0.08, revenue_yoy=0.62, eps=2.46, eps_qoq=0.11,
                   gross_margin=0.732, gross_margin_prior=0.746).sentence()
        assert "$96.22B" in s and "+62% YoY" in s
        assert "not a beat or a miss" in s

    def test_nothing_filed_yields_an_empty_string_not_a_guess(self):
        from datetime import date

        from swing.analysis.earnings import characterize

        assert characterize("ZZZZ", date(2026, 1, 1)) == ""


class TestEarningsFiguresReachThePrompt:
    """Step 1.7 asks the agent to characterise a known catalyst. An instruction
    alone cannot do that — it needs the filed numbers."""

    def test_the_block_carries_the_figures(self, monkeypatch):
        from datetime import date

        from swing.agent import prompts

        monkeypatch.setattr("swing.analysis.earnings.characterize",
                            lambda t, d: "Revenue $96.22B, +62% YoY. Not a beat or a miss.")
        block = prompts._earnings_block("NVDA", date(2026, 8, 27))
        assert "Earnings mode" in block and "$96.22B" in block

    def test_nothing_filed_degrades_to_the_plain_instruction(self, monkeypatch):
        from datetime import date

        from swing.agent import prompts

        monkeypatch.setattr("swing.analysis.earnings.characterize", lambda t, d: "")
        assert prompts._earnings_block("ZZZZ", date(2026, 8, 27)) == prompts.EARNINGS_INSTRUCTION

    def test_an_sec_outage_never_fails_the_attribution(self, monkeypatch):
        """A network error while building a prompt must not lose the answer."""
        from datetime import date

        from swing.agent import prompts

        def boom(*_a, **_k):
            raise ConnectionError("data.sec.gov unreachable")

        monkeypatch.setattr("swing.analysis.earnings.characterize", boom)
        assert prompts._earnings_block("NVDA", date(2026, 8, 27)) == prompts.EARNINGS_INSTRUCTION

    def test_a_non_earnings_swing_gets_no_earnings_text(self):
        import inspect

        from swing.agent import prompts

        src = inspect.getsource(prompts.render)
        assert 'if swing["earnings_mode"] else ""' in src
