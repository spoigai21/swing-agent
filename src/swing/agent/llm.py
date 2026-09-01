"""Provider abstraction for the attribution model.

Swapping providers must be one config line (agent-plan.md 1.1), so nothing
outside this module names a model or a vendor.
"""
from __future__ import annotations

from typing import Any

from swing.common.settings import get_settings


def get_llm(temperature: float = 0.0, **kwargs: Any):
    """The raw chat model.

    temperature=0 is NOT optional: the Phase 4 harness reruns the same placebo
    cases after every prompt change, and with sampling noise you cannot tell
    whether a metric moved because of your edit. agent-plan.md 3.4.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = get_settings()
    settings.require("gemini_api_key")
    model = settings.attribution_model.split(":", 1)[-1]  # tolerate a 'google_genai:' prefix

    # langchain-google-genai reads GOOGLE_API_KEY from the environment, but our
    # key is loaded into Settings from .env and never exported. Pass it
    # explicitly or construction fails with a misleading "API key required".
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=settings.gemini_api_key,
        # Thinking/reasoning is left at the provider default and should be
        # minimised on this node: longer reasoning chains measurably degrade
        # abstention recall, and abstention is what this node exists to get
        # right. agent-plan.md 1.1 / 3.4.
        **kwargs,
    )


def get_attributor(temperature: float = 0.0):
    """The model bound to the Attribution schema via constrained decoding.

    Nested schema verified working 2026-08-31 (see agent/schema.py); the flat
    fallback is not used.
    """
    from swing.agent.schema import Attribution

    return get_llm(temperature=temperature).with_structured_output(Attribution)
