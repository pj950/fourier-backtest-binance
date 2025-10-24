from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from config.settings import settings
from core.data.binance_client import BinanceClient
from core.data.cache import KlineCache, klines_to_dataframe
from core.data.loader import load_klines


def test_klines_to_dataframe() -> None:
    mock_klines = [
        [
            1609459200000,
            "29000.00",
            "29500.00",
            "28800.00",
            "29200.00",
            "100.5",
            1609460999999,
            "2920000.00",
            1500,
            "50.2",
            "1460000.00",
            "0",
        ],
        [
            1609461000000,
            "29200.00",
            "29800.00",
            "29100.00",
            "29600.00",
            "120.3",
            1609462799999,
            "3560000.00",
            1800,
            "60.1",
            "1780000.00",
            "0",
        ],
    ]

    df = klines_to_dataframe(mock_klines)

    assert len(df) == 2
    assert list(df.columns) == [
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
    assert df["open"].dtype in [float, "float64"]
    assert df["trades"].dtype in [int, "int64"]
    assert pd.api.types.is_datetime64_any_dtype(df["open_time"])


def test_klines_to_dataframe_empty() -> None:
    df = klines_to_dataframe([])
    assert df.empty


def test_cache_merge() -> None:
    df1 = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=3, freq="1h", tz=UTC),
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [103.0, 104.0, 105.0],
            "volume": [1000.0, 1100.0, 1200.0],
            "quote_volume": [100000.0, 110000.0, 120000.0],
            "trades": [50, 55, 60],
            "close_time": pd.date_range("2024-01-01 00:59:59", periods=3, freq="1h", tz=UTC),
        }
    )

    df2 = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01 02:00:00", periods=2, freq="1h", tz=UTC),
            "open": [102.0, 103.0],
            "high": [107.0, 108.0],
            "low": [101.0, 102.0],
            "close": [105.0, 106.0],
            "volume": [1200.0, 1300.0],
            "quote_volume": [120000.0, 130000.0],
            "trades": [60, 65],
            "close_time": pd.date_range("2024-01-01 02:59:59", periods=2, freq="1h", tz=UTC),
        }
    )

    cache = KlineCache("TESTUSDT", "1h")
    cache.save(df1)
    result = cache.merge_and_save(df2)

    assert len(result) == 4
    assert result["open_time"].is_monotonic_increasing
    assert not result.duplicated(subset=["open_time"]).any()


def test_gap_detection() -> None:
    times = pd.date_range("2024-01-01", periods=5, freq="1h", tz=UTC).tolist()
    times.append(times[-1] + timedelta(hours=3))

    df = pd.DataFrame(
        {
            "open_time": times,
            "open": [100.0] * 6,
            "high": [105.0] * 6,
            "low": [99.0] * 6,
            "close": [103.0] * 6,
            "volume": [1000.0] * 6,
            "quote_volume": [100000.0] * 6,
            "trades": [50] * 6,
            "close_time": [t + timedelta(minutes=59, seconds=59) for t in times],
        }
    )

    cache = KlineCache("TESTUSDT", "1h")
    cache.save(df)

    gaps = cache.detect_gaps("1h")

    assert len(gaps) >= 1


def test_load_nonexistent_cache() -> None:
    cache = KlineCache("NONEXISTENT", "1h")
    df = cache.load()
    assert df is None


def test_load_klines_validates_inputs() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 2, tzinfo=UTC)

    with pytest.raises(ValueError):
        load_klines("INVALID", "1h", start, end)

    with pytest.raises(ValueError):
        load_klines("BTCUSDT", "15m", start, end)

    with pytest.raises(ValueError):
        load_klines("BTCUSDT", "1h", end, start)


def test_binance_client_paginates(monkeypatch) -> None:
    client = BinanceClient()
    base_ms = 1_700_000_000_000
    step_ms = 3_600_000

    def build_batch(start_ms: int, count: int) -> list[list[int | str]]:
        batch: list[list[int | str]] = []
        for idx in range(count):
            open_ms = start_ms + idx * step_ms
            close_ms = open_ms + step_ms - 1
            batch.append(
                [
                    open_ms,
                    "1.0",
                    "1.0",
                    "1.0",
                    "1.0",
                    "1.0",
                    close_ms,
                    "1.0",
                    1,
                    "0.0",
                    "0.0",
                    "0",
                ]
            )
        return batch

    responses = [
        build_batch(base_ms, 1000),
        build_batch(base_ms + 1000 * step_ms, 50),
    ]
    calls: list[tuple[datetime, datetime]] = []

    def fake_fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        limit: int | None = None,
    ) -> list[list[int | str]]:
        calls.append((start_time, end_time))
        return responses.pop(0)

    monkeypatch.setattr(BinanceClient, "fetch_klines", fake_fetch_klines)

    result = client.fetch_all_klines(
        symbol="BTCUSDT",
        interval="1h",
        start_time=datetime.fromtimestamp(base_ms / 1000, tz=UTC),
        end_time=datetime.fromtimestamp((base_ms + 1_050 * step_ms) / 1000, tz=UTC),
    )

    assert len(result) == 1050
    assert len(calls) == 2
    client.close()


def test_ensure_range_fetches_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "cache_dir", tmp_path)
    cache = KlineCache("TESTUSDT", "1h")

    existing_times = pd.date_range("2024-01-02", periods=5, freq="1h", tz=UTC)
    existing_df = pd.DataFrame(
        {
            "open_time": existing_times,
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.5] * 5,
            "volume": [10.0] * 5,
            "quote_volume": [1000.0] * 5,
            "trades": [1] * 5,
            "close_time": existing_times + timedelta(minutes=59, seconds=59),
        }
    )
    cache.save(existing_df)

    calls: list[tuple[datetime, datetime]] = []

    def build_raw_klines(start_dt: datetime, count: int) -> list[list[int | str]]:
        step_ms = 3_600_000
        base_ms = int(start_dt.timestamp() * 1000)
        batch: list[list[int | str]] = []
        for idx in range(count):
            open_ms = base_ms + idx * step_ms
            close_ms = open_ms + step_ms - 1
            batch.append(
                [
                    open_ms,
                    "1.0",
                    "1.0",
                    "1.0",
                    "1.0",
                    "1.0",
                    close_ms,
                    "1.0",
                    1,
                    "0.0",
                    "0.0",
                    "0",
                ]
            )
        return batch

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 2, 4, tzinfo=UTC)

    def fake_fetch_all(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[list[int | str]]:
        calls.append((start_time, end_time))
        if start_time == start:
            return build_raw_klines(start, 24)
        return []

    monkeypatch.setattr(BinanceClient, "fetch_all_klines", fake_fetch_all)

    result = cache.ensure_range(start_time=start, end_time=end)

    assert calls
    assert calls[0][0] == start
    assert not result.empty
    assert result.iloc[0]["open_time"] == pd.Timestamp(start)
    assert result.iloc[-1]["open_time"] == pd.Timestamp(end)
