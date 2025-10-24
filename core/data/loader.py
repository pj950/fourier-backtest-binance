from __future__ import annotations

from datetime import datetime

import pandas as pd

from core.data.cache import KlineCache

SUPPORTED_SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
SUPPORTED_INTERVALS: tuple[str, ...] = ("30m", "1h", "4h")


def load_klines(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    force_refresh: bool = False,
) -> pd.DataFrame:
    if symbol not in SUPPORTED_SYMBOLS:
        raise ValueError(
            f"Unsupported symbol '{symbol}'. Supported symbols: {', '.join(SUPPORTED_SYMBOLS)}"
        )

    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(
            f"Unsupported interval '{interval}'. Supported intervals: {', '.join(SUPPORTED_INTERVALS)}"
        )

    if start >= end:
        raise ValueError("Start time must be earlier than end time")

    cache = KlineCache(symbol=symbol, interval=interval)

    if force_refresh:
        cache.clear()
        df = cache.fetch_and_update(start_time=start, end_time=end)
    else:
        df = cache.ensure_range(start_time=start, end_time=end)

    if df is None or df.empty:
        return pd.DataFrame()

    df = df[(df["open_time"] >= start) & (df["open_time"] <= end)].reset_index(drop=True)

    return df
