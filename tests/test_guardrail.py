"""Forecast refusal. Must fire BEFORE any model call.

agent-plan.md states the system is not a price predictor and must not be tuned
or evaluated as one; swing-cli.md Part 8 requires the refusal to happen without
spending an LLM call. Deliberately conservative: a false refusal costs one
rephrase, a false accept produces investment advice from a system with no
forecasting capability.
"""
from __future__ import annotations

import pytest

from swing.interface.guardrail import check

REFUSE = [
    "is NVDA a buy?",
    "should I buy TSLA",
    "should i sell my AAPL shares",
    "will NVDA go up next week?",
    "would MU rise after earnings",
    "what is the price target for MU?",
    "give me a forecast for the semis",
    "predict NVDA's move tomorrow",
    "is it a good time to buy AAPL?",
    "what's the outlook for semiconductors",
    "is TTWO worth buying",
    "buy signal for QCOM?",
    "what will happen next quarter",
]

ALLOW = [
    "why did NVDA move on 2026-08-27?",
    "show me every unexplained swing this month",
    "which tickers have the highest idiosyncratic share?",
    "what drove MU last month?",
    "biggest swings this week",
    "what caused the Netflix drop",
    "explain the semiconductor selloff",
    "how many articles do we have",
]


@pytest.mark.parametrize("q", REFUSE)
def test_forward_looking_is_refused(q):
    r = check(q)
    assert not r.allowed, f"should refuse: {q!r}"
    assert "not a price predictor" in (r.reason or "")


@pytest.mark.parametrize("q", ALLOW)
def test_backward_looking_is_allowed(q):
    assert check(q).allowed, f"should allow: {q!r}"


def test_empty_question_is_refused():
    assert not check("").allowed
    assert not check("   ").allowed


def test_refusal_names_what_the_system_can_do():
    # A bare refusal is unhelpful; it must redirect.
    r = check("is NVDA a buy?")
    assert "why did" in (r.reason or "").lower()


def test_guardrail_runs_before_any_model_call(monkeypatch):
    """The whole point: refusing must not spend quota."""
    import swing.interface.query as q

    called = []
    monkeypatch.setattr(q, "_llm_fallback",
                        lambda *a, **k: called.append(1) or q.Answer("x"))
    out = q.answer("should I buy NVDA?")
    assert not called, "guardrail must short-circuit before the LLM fallback"
    assert not out.used_llm
    assert "not a price predictor" in out.text
