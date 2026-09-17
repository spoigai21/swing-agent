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
