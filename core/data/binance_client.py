from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import settings
from core.data.exceptions import (
    BinanceRateLimitError,
    BinanceRequestError,
    BinanceTransientError,
)
from core.utils.time import datetime_to_timestamp, timestamp_to_datetime


class BinanceClient:
    def __init__(self) -> None:
        self.base_url = settings.binance_base_url.rstrip("/")
        self.timeout = settings.binance_request_timeout
        self.limit = settings.binance_klines_limit
        self.client = httpx.Client(timeout=self.timeout)

    def __enter__(self) -> "BinanceClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        self.client.close()

    @retry(
        retry=retry_if_exception_type((BinanceTransientError, BinanceRateLimitError)),
        stop=stop_after_attempt(settings.max_retry_attempts),
        wait=wait_exponential(multiplier=settings.retry_initial_wait, max=settings.retry_max_wait),
        reraise=True,
    )
    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        limit: int | None = None,
    ) -> list[list[Any]]:
        if start_time >= end_time:
            raise BinanceRequestError("start_time must be earlier than end_time")

        effective_limit = limit or self.limit
        start_ms = datetime_to_timestamp(start_time)
        end_ms = datetime_to_timestamp(end_time)

        url = f"{self.base_url}/api/v3/klines"
        params: dict[str, str | int] = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": effective_limit,
        }

        try:
            response = self.client.get(url, params=params)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise BinanceTransientError(
                f"Request to Binance timed out after {self.timeout} seconds"
            ) from exc
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            message = self._extract_error_message(exc)
            if status_code == 429:
                raise BinanceRateLimitError(
                    message,
                    retry_after=self._parse_retry_after(exc.response),
                    used_weight=self._parse_used_weight(exc.response),
                ) from exc
            if 500 <= status_code < 600:
                raise BinanceTransientError(message) from exc
            raise BinanceRequestError(message) from exc
        except httpx.RequestError as exc:
            raise BinanceTransientError(f"Network error connecting to Binance: {exc}") from exc

        result = response.json()
        if not isinstance(result, list):
            raise BinanceRequestError("Unexpected response format from Binance")
        return result

    def fetch_all_klines(
        self, symbol: str, interval: str, start_time: datetime, end_time: datetime
    ) -> list[list[Any]]:
        if start_time >= end_time:
            raise BinanceRequestError("start_time must be earlier than end_time")

        all_klines: list[list[Any]] = []
        current_start = start_time

        while current_start < end_time:
            batch = self.fetch_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_time,
                limit=self.limit,
            )

            if not batch:
                break

            all_klines.extend(batch)

            last_close_raw = batch[-1][6]
            try:
                last_close_time = int(last_close_raw)
            except (TypeError, ValueError) as exc:
                raise BinanceRequestError(f"Invalid close time in Binance response: {last_close_raw}") from exc

            next_start = timestamp_to_datetime(last_close_time + 1)
            if next_start <= current_start:
                break

            current_start = next_start

            if len(batch) < self.limit:
                break

        return all_klines

    @staticmethod
    def _extract_error_message(exc: httpx.HTTPStatusError) -> str:
        response = exc.response
        status_code = response.status_code
        message = f"Binance API error ({status_code})"

        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            detail = payload.get("msg") or payload.get("message")
            if isinstance(detail, str) and detail.strip():
                return f"Binance API error ({status_code}): {detail.strip()}"

        text = response.text.strip()
        if text:
            return f"Binance API error ({status_code}): {text[:200]}"

        return message

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> float | None:
        header = response.headers.get("Retry-After")
        if header is None:
            return None
        try:
            return float(header)
        except ValueError:
            return None

    @staticmethod
    def _parse_used_weight(response: httpx.Response) -> int | None:
        header = response.headers.get("x-mbx-used-weight-1m")
        if header is None:
            return None
        try:
            return int(header)
        except ValueError:
            return None
