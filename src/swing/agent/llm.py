"""Provider abstraction for the attribution model.

⚠️ FREE-TIER DAILY CAPS ARE PER MODEL AND SMALL.

Probed 2026-09-03 with a live key:
  gemini-2.5-flash, gemini-2.5-flash-lite   ModelNotFound (retired)
  gemini-3.7-flash                          429, daily cap = 20 requests
  gemini-3.5-flash, gemini-flash-latest,
  gemini-flash-lite-latest                  available

agent-plan.md 1.1 warned that Pro free tiers ran as low as 50/day; a
Flash-class model at 20/day is tighter still. Production is 1-3 attributions a
day and fits easily, but a Phase 4 placebo sweep is 200 requests and does NOT.
Iterate against the 30-case smoke set and budget full sweeps across days.

Swapping providers must be one config line (agent-plan.md 1.1), so nothing
outside this module names a model or a vendor.
"""
from __future__ import annotations

from typing import Any

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from swing.common.http import _RateLimiter
from swing.common.settings import get_settings

_llm_limiter = _RateLimiter(4.0)   # free tiers throttle per-minute as well as per-day


def _is_transient(exc: BaseException) -> bool:
    """Rate limits (429) and overload (503). Both usually clear quickly."""
    text = str(exc)
    return any(s in text for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))


@retry(
    retry=retry_if_exception(_is_transient),
    # A person is waiting at the prompt: three tries over ~30 s, not minutes.
    # On 2026-09-14 every Flash model returned intermittent 503 "high demand";
    # only 429 was retried, so one spike failed the whole question.
    wait=wait_exponential(multiplier=2, min=5, max=20),
    stop=stop_after_attempt(3),
    reraise=True,
)
def invoke_with_retry(runnable, prompt: str):
    """Paced, 429-aware invoke. agent-plan.md 3.4 requires this.

    Free-tier quotas throttle by requests-per-minute as well as per-day, and a
    batch firing attributions back to back will trip RPM long before the daily
    cap. The batch has no latency requirement, so pacing costs nothing.
    """
    _llm_limiter.wait()
    return runnable.invoke(prompt)


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
        # ⚠️ The client defaults to NO timeout and 6 retries. On 2026-09-14 the
        # pinned model stopped answering and a question hung indefinitely
        # instead of reporting "couldn't reach Gemini". Bounded: at most two
        # 60 s attempts, then llm_attribute records llm_error.
        timeout=60,
        max_retries=1,
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
