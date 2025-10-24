from datetime import datetime

import pandas as pd

from core.data.cache import KlineCache


def load_klines(
    symbol: str, interval: str, start: datetime, end: datetime, force_refresh: bool = False
) -> pd.DataFrame:
    cache = KlineCache(symbol=symbol, interval=interval)

    if force_refresh:
        df = cache.fetch_and_update(start_time=start, end_time=end)
    else:
        cache.incremental_update()
        df = cache.load()

        if df is None or df.empty:
            df = cache.fetch_and_update(start_time=start, end_time=end)

    if df is None or df.empty:
        return pd.DataFrame()

    df = df[(df["open_time"] >= start) & (df["open_time"] <= end)].reset_index(drop=True)

    return df
