"""
Centralized Gemini API retry, rate-limiting, and safety utilities.

Provides:
    - ``gemini_retry`` — tenacity decorator for retrying on 503 / 429 errors
    - ``gemini_rate_limiter`` — async semaphore-based concurrency limiter
    - ``call_gemini`` — wrapper that combines rate-limiting with the API call
    - ``GeminiCircuitBreaker`` — prevents cascading retries when Gemini is down
"""

import asyncio
import logging
import time
from typing import Any

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from ..core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1) Retry predicate — which exceptions should trigger a retry?
# ---------------------------------------------------------------------------

def _is_retryable_error(error: BaseException) -> bool:
    """Return True for 503 (overloaded) and 429 (rate-limit) Gemini errors."""
    msg = str(error).lower()
    return any(indicator in msg for indicator in [
        "503",
        "429",
        "quota",
        "rate limit",
        "rate_limit",
        "resource exhausted",
        "too many requests",
        "overloaded",
        "unavailable",
        "high demand",
        "service unavailable",
    ])


# ---------------------------------------------------------------------------
# 2) Circuit breaker — prevents cascading retries across pipeline stages
# ---------------------------------------------------------------------------

class GeminiCircuitBreaker:
    """
    When one call_gemini exhausts all retries, the circuit opens and
    subsequent calls fail immediately for ``cooldown`` seconds.
    This prevents a 5-stage pipeline from each spending 20s retrying
    when Gemini is clearly down.
    """

    def __init__(self, cooldown: float = 30.0):
        self._cooldown = cooldown
        self._open_until: float = 0.0          # monotonic timestamp
        self._last_error: BaseException | None = None

    @property
    def is_open(self) -> bool:
        return time.monotonic() < self._open_until

    def trip(self, error: BaseException) -> None:
        """Open the circuit after retries exhausted."""
        self._open_until = time.monotonic() + self._cooldown
        self._last_error = error
        logger.warning(
            f"Circuit breaker OPEN for {self._cooldown}s — "
            f"Gemini unavailable: {error}"
        )

    def check(self) -> None:
        """Raise immediately if the circuit is open."""
        if self.is_open and self._last_error is not None:
            logger.info("Circuit breaker is OPEN — skipping Gemini call")
            raise self._last_error

    def reset(self) -> None:
        """Close the circuit (successful call)."""
        if self._open_until > 0:
            logger.info("Circuit breaker CLOSED — Gemini is back")
        self._open_until = 0.0
        self._last_error = None


# Global singleton
circuit_breaker = GeminiCircuitBreaker(cooldown=30.0)


# ---------------------------------------------------------------------------
# 3) Tenacity retry decorator — reusable across all Gemini call sites
# ---------------------------------------------------------------------------

gemini_retry = retry(
    retry=retry_if_exception(_is_retryable_error),
    stop=stop_after_attempt(settings.GEMINI_MAX_RETRIES),      # default 3
    wait=wait_exponential(
        multiplier=settings.GEMINI_RETRY_DELAY,                # default 2
        min=settings.GEMINI_RETRY_DELAY,
        max=10,                                                # cap at 10s
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


# ---------------------------------------------------------------------------
# 4) Async rate limiter — limits concurrent Gemini calls
# ---------------------------------------------------------------------------

class GeminiRateLimiter:
    """
    Controls Gemini API usage with:
      - Concurrency cap (asyncio.Semaphore)
      - Minimum delay between consecutive calls (token bucket)
    """

    def __init__(
        self,
        max_concurrent: int = 2,
        min_interval: float = 1.0,
    ):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._min_interval = min_interval
        self._last_call: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a call slot is available and enough time has passed."""
        await self._semaphore.acquire()
        async with self._lock:
            now = time.monotonic()
            wait_time = self._min_interval - (now - self._last_call)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._last_call = time.monotonic()

    def release(self) -> None:
        self._semaphore.release()


# Global singleton — shared across all services
gemini_rate_limiter = GeminiRateLimiter(max_concurrent=2, min_interval=1.0)


# ---------------------------------------------------------------------------
# 5) Convenience wrapper — circuit breaker + rate-limit + retry in one
# ---------------------------------------------------------------------------

@gemini_retry
async def call_gemini(
    client: Any,
    model: str,
    contents: Any,
    config: Any,
) -> Any:
    """
    Call Gemini generate_content with circuit breaker, rate-limiting,
    and automatic retry.

    Raises the original exception after ``GEMINI_MAX_RETRIES`` attempts
    so callers can decide what to do (instead of silently returning None).
    """
    # Fast-fail if a previous call already proved Gemini is down
    circuit_breaker.check()

    await gemini_rate_limiter.acquire()
    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        # Success — close the circuit if it was open
        circuit_breaker.reset()
        return response
    except Exception as exc:
        # If this is a retryable error and we're about to exhaust retries,
        # tenacity will re-raise. The caller below will trip the breaker.
        raise
    finally:
        gemini_rate_limiter.release()


async def call_gemini_safe(
    client: Any,
    model: str,
    contents: Any,
    config: Any,
) -> Any:
    """
    Top-level wrapper that trips the circuit breaker after retries exhaust.
    Use this instead of call_gemini directly.
    """
    try:
        return await call_gemini(client, model, contents, config)
    except Exception as exc:
        if _is_retryable_error(exc):
            circuit_breaker.trip(exc)
        raise
    finally:
        gemini_rate_limiter.release()
