from datetime import datetime
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from core.utils.time import datetime_to_timestamp, timestamp_to_datetime


class BinanceClient:
    def __init__(self) -> None:
        self.base_url = settings.binance_base_url
        self.timeout = settings.binance_request_timeout
        self.client = httpx.Client(timeout=self.timeout)

    def __enter__(self) -> "BinanceClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        self.client.close()

    @retry(
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
        limit: int = 1000,
    ) -> list[list[Any]]:
        start_ms = datetime_to_timestamp(start_time)
        end_ms = datetime_to_timestamp(end_time)

        url = f"{self.base_url}/api/v3/klines"
        params: dict[str, str | int] = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }

        response = self.client.get(url, params=params)
        response.raise_for_status()

        result: list[list[Any]] = response.json()
        return result

    def fetch_all_klines(
        self, symbol: str, interval: str, start_time: datetime, end_time: datetime
    ) -> list[list[Any]]:
        all_klines: list[list[Any]] = []
        current_start = start_time

        while current_start < end_time:
            klines = self.fetch_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_time,
                limit=1000,
            )

            if not klines:
                break

            all_klines.extend(klines)

            last_close_time = klines[-1][6]
            current_start = timestamp_to_datetime(last_close_time + 1)

            if len(klines) < 1000:
                break

        return all_klines
