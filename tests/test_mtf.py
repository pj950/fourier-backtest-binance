import numpy as np
import pandas as pd
import pytest

from core.analysis.mtf import (
    align_timeframes,
    apply_mtf_filter,
    check_mtf_alignment,
    compute_trend_direction,
)


def test_compute_trend_direction_uptrend():
    close = np.array([100, 102, 105, 108, 110])
    smoothed = np.array([99, 100, 101, 103, 105])

    trend = compute_trend_direction(close, smoothed, slope_lookback=1)

    assert trend[0] == 1
    assert trend[-1] == 1
    assert np.all((trend == 1) | (trend == 0))


def test_compute_trend_direction_downtrend():
    close = np.array([110, 108, 105, 102, 100])
    smoothed = np.array([112, 110, 108, 106, 104])

    trend = compute_trend_direction(close, smoothed, slope_lookback=1)

    assert trend[-1] == -1
    assert np.any(trend == -1)


def test_compute_trend_direction_neutral():
    close = np.array([100, 101, 100, 101, 100])
    smoothed = np.array([100, 100, 100, 100, 100])

    trend = compute_trend_direction(close, smoothed, slope_lookback=1)

    assert np.all((trend == 0) | (trend == 1) | (trend == -1))


def test_compute_trend_direction_empty():
    close = np.array([])
    smoothed = np.array([])

    trend = compute_trend_direction(close, smoothed)

    assert len(trend) == 0


def test_check_mtf_alignment_all_long():
    trend_30m = np.array([1, 1, 1, 0, -1])
    trend_1h = np.array([1, 1, 1, 1, 0])
    trend_4h = np.array([1, 1, 1, 1, 1])

    aligned_long, aligned_short = check_mtf_alignment(
        trend_30m, trend_1h, trend_4h, require_all=True
    )

    assert aligned_long[0] == True
    assert aligned_long[1] == True
    assert aligned_long[2] == True
    assert aligned_long[3] == False
    assert aligned_long[4] == False

    assert np.all(aligned_short == False)


def test_check_mtf_alignment_all_short():
    trend_30m = np.array([-1, -1, -1, 0, 1])
    trend_1h = np.array([-1, -1, -1, -1, 0])
    trend_4h = np.array([-1, -1, -1, -1, -1])

    aligned_long, aligned_short = check_mtf_alignment(
        trend_30m, trend_1h, trend_4h, require_all=True
    )

    assert aligned_short[0] == True
    assert aligned_short[1] == True
    assert aligned_short[2] == True
    assert aligned_short[3] == False

    assert np.all(aligned_long == False)


def test_check_mtf_alignment_majority():
    trend_30m = np.array([1, 1, 1, 0, -1])
    trend_1h = np.array([1, 1, 0, 0, 0])
    trend_4h = np.array([1, 0, 1, 1, 1])

    aligned_long, aligned_short = check_mtf_alignment(
        trend_30m, trend_1h, trend_4h, require_all=False
    )

    assert aligned_long[0] == True
    assert aligned_long[1] == True
    assert aligned_long[2] == True


def test_check_mtf_alignment_empty():
    trend_30m = np.array([])
    trend_1h = np.array([])
    trend_4h = np.array([])

    aligned_long, aligned_short = check_mtf_alignment(
        trend_30m, trend_1h, trend_4h
    )

    assert len(aligned_long) == 0
    assert len(aligned_short) == 0


def test_apply_mtf_filter_long():
    signals = np.array([1, 0, 1, 0, -1, 1, 0])
    mtf_aligned = np.array([True, True, False, False, False, True, True])

    filtered = apply_mtf_filter(signals, mtf_aligned, direction=1)

    assert filtered[0] == 1
    assert filtered[2] == 0
    assert filtered[4] == -1
    assert filtered[5] == 1


def test_apply_mtf_filter_short():
    signals = np.array([0, -1, 0, -1, 1, 0])
    mtf_aligned = np.array([False, True, True, False, False, True])

    filtered = apply_mtf_filter(signals, mtf_aligned, direction=-1)

    assert filtered[1] == -1
    assert filtered[3] == 0


def test_apply_mtf_filter_empty():
    signals = np.array([])
    mtf_aligned = np.array([], dtype=bool)

    filtered = apply_mtf_filter(signals, mtf_aligned)

    assert len(filtered) == 0


def test_align_timeframes():
    df_30m = pd.DataFrame({
        'open_time': pd.date_range('2024-01-01 00:00', periods=4, freq='30min'),
        'close': [100, 101, 102, 103],
    })

    df_1h = pd.DataFrame({
        'open_time': pd.date_range('2024-01-01 00:00', periods=2, freq='1h'),
        'close': [100.5, 102.5],
        'high': [101, 103],
    })

    result = align_timeframes(df_30m, df_1h, '30m', '1h')

    assert len(result) == 4
    assert 'close_1h' in result.columns
    assert 'high_1h' in result.columns
    assert result['close_1h'].iloc[0] == 100.5
    assert result['close_1h'].iloc[1] == 100.5
    assert result['close_1h'].iloc[2] == 102.5
    assert result['close_1h'].iloc[3] == 102.5


def test_align_timeframes_empty():
    df_30m = pd.DataFrame({
        'open_time': pd.date_range('2024-01-01 00:00', periods=4, freq='30min'),
        'close': [100, 101, 102, 103],
    })

    df_1h = pd.DataFrame()

    result = align_timeframes(df_30m, df_1h, '30m', '1h')

    assert len(result) == 4
    assert result.equals(df_30m)
