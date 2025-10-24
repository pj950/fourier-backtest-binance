from __future__ import annotations

from datetime import UTC

import pandas as pd

from core.data.cache import KlineCache
from core.data.streaming import BinanceKlineStreamer


def _ms(timestamp: pd.Timestamp) -> int:
    return int(timestamp.timestamp() * 1000)


def test_streamer_updates_snapshot(monkeypatch) -> None:
    times = pd.date_range("2024-01-01", periods=2, freq="30T", tz=UTC)
    close_times = times + pd.Timedelta(minutes=30) - pd.Timedelta(milliseconds=1)
    initial_df = pd.DataFrame(
        {
            "open_time": times,
            "open": [100.0, 105.0],
            "high": [101.0, 106.0],
            "low": [99.0, 104.0],
            "close": [100.5, 105.5],
            "volume": [10.0, 11.0],
            "quote_volume": [1000.0, 1100.0],
            "trades": [100, 120],
            "close_time": close_times,
        }
    )

    cache = KlineCache("TESTUSDT", "30m")
    monkeypatch.setattr(cache, "merge_and_save", lambda df: df)

    streamer = BinanceKlineStreamer(
        "TESTUSDT",
        "30m",
        cache=cache,
        start_time=times.min().to_pydatetime(),
        initial_data=initial_df,
    )

    merge_calls: list[pd.DataFrame] = []

    def fake_merge(df: pd.DataFrame) -> pd.DataFrame:
        merge_calls.append(df.copy())
        return df

    monkeypatch.setattr(streamer._cache, "merge_and_save", fake_merge)

    last_open = times.iloc[-1]
    last_close = close_times.iloc[-1]

    partial_payload = {
        "t": _ms(last_open),
        "T": _ms(last_close),
        "o": "105.0",
        "h": "107.0",
        "l": "103.0",
        "c": "106.0",
        "v": "12.0",
        "q": "1300.0",
        "n": 150,
        "x": False,
    }

    streamer._handle_kline_payload(partial_payload)
    snapshot = streamer.snapshot()
    assert snapshot.iloc[-1]["close"] == 106.0

    updates = streamer.drain_updates()
    assert len(updates) == 1
    assert not updates[0].is_final
    assert updates[0].source == "ws"
    assert merge_calls == []

    new_open = last_open + pd.Timedelta(minutes=30)
    new_close = new_open + pd.Timedelta(minutes=30) - pd.Timedelta(milliseconds=1)

    final_payload = {
        "t": _ms(new_open),
        "T": _ms(new_close),
        "o": "106.0",
        "h": "108.0",
        "l": "105.0",
        "c": "107.5",
        "v": "13.0",
        "q": "1400.0",
        "n": 160,
        "x": True,
    }

    streamer._handle_kline_payload(final_payload)

    updates = streamer.drain_updates()
    assert updates[-1].is_final
    assert updates[-1].source == "ws"
    assert merge_calls, "merge_and_save should be invoked for final candles"

    snapshot_final = streamer.snapshot()
    assert snapshot_final.iloc[-1]["open_time"] == new_open
    assert snapshot_final.iloc[-1]["close"] == 107.5
    assert len(snapshot_final) == len(initial_df) + 1
