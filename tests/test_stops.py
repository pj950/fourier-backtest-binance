import numpy as np
import pytest

from core.analysis.stops import (
    compute_atr,
    compute_atr_stops,
    compute_residual_quantile,
    compute_residual_sigma,
    compute_residual_stops,
    compute_trailing_stop,
)


def test_compute_atr_basic() -> None:
    """Test basic ATR computation."""
    high = np.array([10, 12, 11, 13, 14])
    low = np.array([8, 10, 9, 11, 12])
    close = np.array([9, 11, 10, 12, 13])

    atr = compute_atr(high, low, close, period=3)

    assert len(atr) == len(high)
    assert np.all(atr >= 0)
    assert not np.any(np.isnan(atr))


def test_compute_atr_empty() -> None:
    """Test ATR with empty arrays."""
    high = np.array([])
    low = np.array([])
    close = np.array([])

    atr = compute_atr(high, low, close)

    assert len(atr) == 0


def test_compute_residual_sigma() -> None:
    """Test residual sigma computation."""
    prices = np.array([100, 102, 98, 101, 99, 103, 97, 102])
    smoothed = np.array([100, 100, 100, 100, 100, 100, 100, 100])

    sigma = compute_residual_sigma(prices, smoothed, window=3)

    assert len(sigma) == len(prices)
    assert np.all(sigma >= 0)
    assert not np.any(np.isnan(sigma))


def test_compute_residual_quantile() -> None:
    """Test residual quantile computation."""
    prices = np.array([100, 102, 98, 101, 99, 103, 97, 102])
    smoothed = np.array([100, 100, 100, 100, 100, 100, 100, 100])

    quant = compute_residual_quantile(prices, smoothed, window=3, quantile=0.95)

    assert len(quant) == len(prices)
    assert np.all(quant >= 0)
    assert not np.any(np.isnan(quant))


def test_compute_atr_stops() -> None:
    """Test ATR-based stop computation."""
    close = np.array([100, 102, 98, 101, 99, 103, 97, 102])
    high = np.array([101, 103, 99, 102, 100, 104, 98, 103])
    low = np.array([99, 101, 97, 100, 98, 102, 96, 101])

    long_stop, long_profit, short_stop, short_profit = compute_atr_stops(
        close, high, low, atr_period=3, k_stop=2.0, k_profit=3.0
    )

    assert len(long_stop) == len(close)
    assert len(long_profit) == len(close)
    assert len(short_stop) == len(close)
    assert len(short_profit) == len(close)

    assert np.all(long_stop < close)
    assert np.all(long_profit > close)
    assert np.all(short_stop > close)
    assert np.all(short_profit < close)


def test_compute_residual_stops_sigma() -> None:
    """Test residual-based stops with sigma method."""
    close = np.array([100, 102, 98, 101, 99, 103, 97, 102])
    smoothed = np.array([100, 100, 100, 100, 100, 100, 100, 100])

    long_stop, long_profit, short_stop, short_profit = compute_residual_stops(
        close, smoothed, method="sigma", window=3, k_stop=2.0, k_profit=3.0
    )

    assert len(long_stop) == len(close)
    assert len(long_profit) == len(close)


def test_compute_residual_stops_quantile() -> None:
    """Test residual-based stops with quantile method."""
    close = np.array([100, 102, 98, 101, 99, 103, 97, 102])
    smoothed = np.array([100, 100, 100, 100, 100, 100, 100, 100])

    long_stop, long_profit, short_stop, short_profit = compute_residual_stops(
        close, smoothed, method="quantile", window=3, quantile=0.95, k_stop=2.0, k_profit=3.0
    )

    assert len(long_stop) == len(close)
    assert len(long_profit) == len(close)


def test_compute_residual_stops_invalid_method() -> None:
    """Test that invalid method raises error."""
    close = np.array([100, 102, 98, 101, 99])
    smoothed = np.array([100, 100, 100, 100, 100])

    with pytest.raises(ValueError):
        compute_residual_stops(close, smoothed, method="invalid")


def test_compute_trailing_stop_long() -> None:
    """Test trailing stop for long position."""
    prices = np.array([100, 102, 104, 103, 105, 104, 102])
    entry_price = 100.0
    entry_idx = 0
    direction = 1
    stop_levels = np.array([95, 96, 97, 98, 99, 100, 101])

    trailing = compute_trailing_stop(prices, entry_price, entry_idx, direction, stop_levels)

    assert len(trailing) == len(prices)
    assert trailing[entry_idx] == stop_levels[entry_idx]

    for i in range(entry_idx + 1, len(trailing)):
        assert trailing[i] >= trailing[i - 1]


def test_compute_trailing_stop_short() -> None:
    """Test trailing stop for short position."""
    prices = np.array([100, 98, 96, 97, 95, 96, 98])
    entry_price = 100.0
    entry_idx = 0
    direction = -1
    stop_levels = np.array([105, 104, 103, 102, 101, 100, 99])

    trailing = compute_trailing_stop(prices, entry_price, entry_idx, direction, stop_levels)

    assert len(trailing) == len(prices)
    assert trailing[entry_idx] == stop_levels[entry_idx]

    for i in range(entry_idx + 1, len(trailing)):
        assert trailing[i] <= trailing[i - 1]


def test_compute_trailing_stop_empty() -> None:
    """Test trailing stop with empty array."""
    prices = np.array([])
    entry_price = 100.0
    entry_idx = 0
    direction = 1
    stop_levels = np.array([])

    trailing = compute_trailing_stop(prices, entry_price, entry_idx, direction, stop_levels)

    assert len(trailing) == 0
