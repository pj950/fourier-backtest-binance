import numpy as np

from core.analysis.signals import (
    compute_slope,
    detect_cross_above,
    detect_cross_below,
    filter_signals_by_period,
    generate_signals_with_stops,
    generate_trend_following_signals,
)


def test_compute_slope() -> None:
    """Test slope computation."""
    signal = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    slope = compute_slope(signal, lookback=1)

    assert len(slope) == len(signal)
    assert np.isnan(slope[0])
    assert np.allclose(slope[1:], [1.0, 2.0, 3.0, 4.0])


def test_detect_cross_above() -> None:
    """Test cross above detection."""
    signal = np.array([1.0, 2.0, 3.0, 2.0, 3.0])
    reference = np.array([2.5, 2.5, 2.5, 2.5, 2.5])

    cross = detect_cross_above(signal, reference)

    assert len(cross) == len(signal)
    assert cross[2] == True
    assert cross[4] == True
    assert not cross[0]
    assert not cross[1]


def test_detect_cross_below() -> None:
    """Test cross below detection."""
    signal = np.array([3.0, 2.0, 1.0, 2.0, 1.0])
    reference = np.array([1.5, 1.5, 1.5, 1.5, 1.5])

    cross = detect_cross_below(signal, reference)

    assert len(cross) == len(signal)
    assert cross[2] == True
    assert cross[4] == True
    assert not cross[0]
    assert not cross[1]


def test_detect_cross_empty() -> None:
    """Test cross detection with empty arrays."""
    signal = np.array([])
    reference = np.array([])

    cross_above = detect_cross_above(signal, reference)
    cross_below = detect_cross_below(signal, reference)

    assert len(cross_above) == 0
    assert len(cross_below) == 0


def test_generate_trend_following_signals_basic() -> None:
    """Test basic trend following signal generation."""
    close = np.array([9, 10, 11, 12, 13, 12, 11, 10, 11])
    smoothed = np.array([10, 10.5, 11, 11.5, 12, 11.5, 11, 10.5, 10])

    entry, exit_signal = generate_trend_following_signals(
        close, smoothed, slope_threshold=0.0, slope_lookback=1
    )

    assert len(entry) == len(close)
    assert len(exit_signal) == len(close)
    assert np.sum(entry) >= 0
    assert np.sum(exit_signal) >= 0


def test_generate_trend_following_signals_with_volatility_filter() -> None:
    """Test signal generation with volatility filter."""
    close = np.array([10, 11, 12, 13, 12, 11, 10, 11, 12])
    smoothed = np.array([10, 10, 10, 10, 11, 11, 11, 11, 11])
    volatility = np.array([0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 0.5, 1.5, 1.5])

    entry, exit_signal = generate_trend_following_signals(
        close, smoothed, min_volatility=1.0, volatility=volatility
    )

    assert len(entry) == len(close)

    for i in range(len(entry)):
        if entry[i]:
            assert volatility[i] >= 1.0


def test_generate_trend_following_signals_empty() -> None:
    """Test signal generation with empty arrays."""
    close = np.array([])
    smoothed = np.array([])

    entry, exit_signal = generate_trend_following_signals(close, smoothed)

    assert len(entry) == 0
    assert len(exit_signal) == 0


def test_generate_signals_with_stops() -> None:
    """Test signal generation incorporating stops."""
    close = np.array([10, 11, 12, 13, 12, 11, 10, 11, 12])
    smoothed = np.array([10, 10, 10, 10, 11, 11, 11, 11, 11])
    stop_levels = np.array([9, 9, 9, 9, 10, 10, 10, 10, 10])

    signals = generate_signals_with_stops(close, smoothed, stop_levels)

    assert len(signals) == len(close)
    assert np.all((signals == -1) | (signals == 0) | (signals == 1))


def test_generate_signals_with_stops_stop_exit() -> None:
    """Test that signals correctly exit on stop hit."""
    close = np.array([100, 102, 98, 96, 94])
    smoothed = np.array([100, 101, 102, 103, 104])
    stop_levels = np.array([97, 97, 97, 97, 97])

    signals = generate_signals_with_stops(close, smoothed, stop_levels)

    stop_hit_indices = np.where(close < stop_levels)[0]
    for idx in stop_hit_indices:
        assert signals[idx] == -1


def test_filter_signals_by_period() -> None:
    """Test signal filtering by minimum period."""
    signals = np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 1])
    min_bars = 3

    filtered = filter_signals_by_period(signals, min_bars)

    assert len(filtered) == len(signals)

    entry_indices = np.where(filtered == 1)[0]
    for i in range(1, len(entry_indices)):
        assert entry_indices[i] - entry_indices[i - 1] >= min_bars


def test_filter_signals_by_period_no_filtering() -> None:
    """Test that well-spaced signals are not filtered."""
    signals = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    min_bars = 3

    filtered = filter_signals_by_period(signals, min_bars)

    assert np.array_equal(signals, filtered)


def test_filter_signals_empty() -> None:
    """Test signal filtering with empty array."""
    signals = np.array([])
    filtered = filter_signals_by_period(signals, 3)

    assert len(filtered) == 0
