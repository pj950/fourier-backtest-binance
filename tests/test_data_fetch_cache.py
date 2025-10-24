from datetime import UTC, timedelta

import pandas as pd

from core.data.cache import KlineCache, klines_to_dataframe


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
