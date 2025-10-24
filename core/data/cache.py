from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import settings
from core.data.binance_client import BinanceClient
from core.utils.time import now_utc


def klines_to_dataframe(klines: list[list[Any]]) -> pd.DataFrame:
    if not klines:
        return pd.DataFrame()

    df = pd.DataFrame(
        klines,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["trades"] = df["trades"].astype(int)

    df = df[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trades",
            "close_time",
        ]
    ]

    return df.sort_values("open_time").reset_index(drop=True)


class KlineCache:
    def __init__(self, symbol: str, interval: str) -> None:
        self.symbol = symbol
        self.interval = interval
        self.cache_path = self._get_cache_path()

    def _get_cache_path(self) -> Path:
        return settings.cache_dir / f"{self.symbol}_{self.interval}.parquet"

    def load(self) -> pd.DataFrame | None:
        if not self.cache_path.exists():
            return None

        try:
            df = pd.read_parquet(self.cache_path)
            return df
        except Exception:
            return None

    def save(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        df = df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")
        df.to_parquet(self.cache_path, index=False, engine="pyarrow")

    def get_last_timestamp(self) -> datetime | None:
        df = self.load()
        if df is None or df.empty:
            return None

        max_time: datetime = pd.Timestamp(df["close_time"].max()).to_pydatetime()
        return max_time

    def merge_and_save(self, new_df: pd.DataFrame) -> pd.DataFrame:
        existing_df = self.load()

        if existing_df is None or existing_df.empty:
            self.save(new_df)
            return new_df

        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = (
            combined_df.sort_values("open_time")
            .drop_duplicates(subset=["open_time"], keep="last")
            .reset_index(drop=True)
        )

        self.save(combined_df)
        return combined_df

    def detect_gaps(self, interval: str) -> list[tuple[datetime, datetime]]:
        df = self.load()
        if df is None or df.empty or len(df) < 2:
            return []

        interval_map = {
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
        }

        expected_delta = interval_map.get(interval)
        if expected_delta is None:
            return []

        gaps: list[tuple[datetime, datetime]] = []

        for i in range(len(df) - 1):
            current_close = pd.Timestamp(df.iloc[i]["close_time"]).to_pydatetime()
            next_open = pd.Timestamp(df.iloc[i + 1]["open_time"]).to_pydatetime()

            time_diff = next_open - current_close

            if time_diff > timedelta(milliseconds=1000):
                gaps.append((current_close, next_open))

        return gaps

    def fetch_and_update(
        self, start_time: datetime, end_time: datetime | None = None
    ) -> pd.DataFrame:
        if end_time is None:
            end_time = now_utc()

        with BinanceClient() as client:
            klines = client.fetch_all_klines(
                symbol=self.symbol, interval=self.interval, start_time=start_time, end_time=end_time
            )

        new_df = klines_to_dataframe(klines)
        return self.merge_and_save(new_df)

    def incremental_update(self) -> pd.DataFrame:
        last_ts = self.get_last_timestamp()

        if last_ts is None:
            return self.fetch_and_update(
                start_time=datetime(2020, 1, 1, tzinfo=UTC), end_time=now_utc()
            )

        start_time = last_ts + timedelta(milliseconds=1)
        return self.fetch_and_update(start_time=start_time, end_time=now_utc())

    def fill_gaps(self) -> pd.DataFrame:
        gaps = self.detect_gaps(self.interval)

        if not gaps:
            df = self.load()
            return df if df is not None else pd.DataFrame()

        for gap_start, gap_end in gaps:
            self.fetch_and_update(
                start_time=gap_start + timedelta(milliseconds=1), end_time=gap_end
            )

        df = self.load()
        return df if df is not None else pd.DataFrame()
