from __future__ import annotations

from typing import Any


class BinanceError(RuntimeError):
    """Base exception for Binance client errors."""


class BinanceRequestError(BinanceError):
    """Raised for non-retryable request issues (e.g. bad parameters)."""


class BinanceTransientError(BinanceError):
    """Raised for retryable transient failures."""


class BinanceRateLimitError(BinanceTransientError):
    """Raised when Binance responds with a rate limit error."""

    def __init__(self, message: str, *, retry_after: float | None = None, used_weight: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.used_weight = used_weight

    def context(self) -> dict[str, Any]:
        return {
            "retry_after": self.retry_after,
            "used_weight": self.used_weight,
        }
