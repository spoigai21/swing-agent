"""HTTP clients with per-provider rate limiting and retry.

Every free tier here throttles differently, and a silent failure that leaves a
gap in the record is unrecoverable for news. Each provider gets its own client
with its own minimum inter-request interval.

SEC: 10 req/s per IP; exceeding it earns a temporary block, not just a 429.
     data-sources.md D.2 rule 2. We use a 120ms floor for headroom.
"""
from __future__ import annotations

import threading
import time

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from swing.common.settings import get_settings

SEC_MIN_INTERVAL = 0.12   # seconds between SEC requests (limit is 10/s)
RSS_MIN_INTERVAL = 0.25
TIINGO_MIN_INTERVAL = 1.2   # Tiingo free tier 429s on bursts; pace deliberately
FINNHUB_MIN_INTERVAL = 1.1  # free tier is 60 calls/min


class _RateLimiter:
    """Process-wide minimum interval between calls. Thread-safe."""

    def __init__(self, min_interval: float) -> None:
        self._min = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            delta = time.monotonic() - self._last
            if delta < self._min:
                time.sleep(self._min - delta)
            self._last = time.monotonic()


_sec_limiter = _RateLimiter(SEC_MIN_INTERVAL)
_rss_limiter = _RateLimiter(RSS_MIN_INTERVAL)
_tiingo_limiter = _RateLimiter(TIINGO_MIN_INTERVAL)
_finnhub_limiter = _RateLimiter(FINNHUB_MIN_INTERVAL)

RETRYABLE = (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError)


def sec_headers() -> dict[str, str]:
    return {
        "User-Agent": get_settings().sec_user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }


def _raise_for_retryable(resp: httpx.Response) -> httpx.Response:
    # 429 and 5xx are worth retrying; 403/404 are not (they mean a bad header
    # or a bad CIK, and retrying a 403 extends an SEC block).
    if resp.status_code == 429 or resp.status_code >= 500:
        resp.raise_for_status()
    return resp


@retry(
    retry=retry_if_exception_type(RETRYABLE),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def sec_get(url: str, timeout: float = 30.0) -> httpx.Response:
    """GET against sec.gov / data.sec.gov with the mandatory UA and rate limit."""
    _sec_limiter.wait()
    headers = sec_headers()
    # The Host header must match the actual host, and www.sec.gov != data.sec.gov.
    headers["Host"] = httpx.URL(url).host
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url, headers=headers)
    return _raise_for_retryable(resp)


@retry(
    retry=retry_if_exception_type(RETRYABLE),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def feed_get(url: str, timeout: float = 20.0) -> httpx.Response:
    """GET an RSS/Atom feed. Some publishers 403 a default python UA."""
    _rss_limiter.wait()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) SwingAgent/0.1 research"
        ),
        "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.8",
    }
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url, headers=headers)
    return _raise_for_retryable(resp)


@retry(
    retry=retry_if_exception_type(RETRYABLE),
    wait=wait_exponential(multiplier=2, min=5, max=120),
    stop=stop_after_attempt(4),
    reraise=True,
)
def tiingo_get(url: str, params: dict, timeout: float = 60.0) -> httpx.Response:
    """GET against Tiingo with pacing and 429 backoff.

    The free tier 429s on bursts. A historical onset backfill fires hundreds of
    requests, and without this it silently returns zero bars — swings then fall
    back to the 48h window and the failure looks like missing data rather than
    throttling.
    """
    _tiingo_limiter.wait()
    from swing.common.settings import get_settings

    headers = {"Authorization": f"Token {get_settings().tiingo_api_key}"}
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url, params=params, headers=headers)
    return _raise_for_retryable(resp)


@retry(
    retry=retry_if_exception_type(RETRYABLE),
    wait=wait_exponential(multiplier=2, min=5, max=120),
    stop=stop_after_attempt(5),
    reraise=True,
)
def finnhub_get(path: str, params: dict, timeout: float = 30.0) -> httpx.Response:
    """GET against Finnhub with pacing and 429 backoff.

    The free tier allows 60 calls/min. A news backfill fires hundreds of
    requests; without pacing it 429s and the chunks are silently skipped, which
    looks like "no news existed then" rather than throttling.
    """
    _finnhub_limiter.wait()
    from swing.common.settings import get_settings

    settings = get_settings()
    settings.require("finnhub_api_key")
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get("https://finnhub.io/api/v1" + path,
                          params={**params, "token": settings.finnhub_api_key})
    return _raise_for_retryable(resp)
