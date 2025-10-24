import numpy as np

from core.analysis.fourier import (
    dct_lowpass_smooth,
    estimate_cutoff_from_period,
    smooth_price_series,
)


def test_dct_lowpass_smooth_basic() -> None:
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    smoothed = dct_lowpass_smooth(signal, cutoff_freq=0.5)

    assert len(smoothed) == len(signal)
    assert not np.array_equal(smoothed, signal)


def test_dct_lowpass_smooth_empty() -> None:
    signal = np.array([])
    smoothed = dct_lowpass_smooth(signal, cutoff_freq=0.5)

    assert len(smoothed) == 0


def test_dct_lowpass_smooth_removes_noise() -> None:
    t = np.linspace(0, 10, 200)
    clean_signal = np.sin(2 * np.pi * 0.5 * t)
    noise = np.random.normal(0, 0.1, len(t))
    noisy_signal = clean_signal + noise

    smoothed = dct_lowpass_smooth(noisy_signal, cutoff_freq=0.2)

    noise_variance = np.var(noisy_signal - clean_signal)
    smoothed_variance = np.var(smoothed - clean_signal)

    assert smoothed_variance < noise_variance


def test_dct_lowpass_smooth_edge_artifacts() -> None:
    signal = np.sin(np.linspace(0, 4 * np.pi, 100))
    smoothed = dct_lowpass_smooth(signal, cutoff_freq=0.5, padding_ratio=0.2)

    edge_region = 10
    start_diff = np.abs(smoothed[:edge_region] - signal[:edge_region])
    end_diff = np.abs(smoothed[-edge_region:] - signal[-edge_region:])

    assert np.mean(start_diff) < 0.5
    assert np.mean(end_diff) < 0.5


def test_dct_lowpass_smooth_preserves_dc() -> None:
    signal = np.ones(50) * 5.0
    smoothed = dct_lowpass_smooth(signal, cutoff_freq=0.5)

    assert np.allclose(smoothed, signal, rtol=0.01)


def test_dct_lowpass_smooth_tapered_cutoff() -> None:
    signal = np.random.randn(100)

    smoothed_sharp = dct_lowpass_smooth(signal, cutoff_freq=0.3, taper_width=0.0)
    smoothed_tapered = dct_lowpass_smooth(signal, cutoff_freq=0.3, taper_width=0.3)

    assert not np.array_equal(smoothed_sharp, smoothed_tapered)


def test_estimate_cutoff_from_period() -> None:
    cutoff = estimate_cutoff_from_period(min_period_bars=10, total_bars=100)

    assert 0.0 < cutoff <= 1.0


def test_estimate_cutoff_from_period_edge_cases() -> None:
    cutoff_zero = estimate_cutoff_from_period(min_period_bars=0, total_bars=100)
    assert cutoff_zero == 1.0

    cutoff_large = estimate_cutoff_from_period(min_period_bars=1000, total_bars=100)
    assert 0.0 < cutoff_large <= 1.0


def test_estimate_cutoff_from_period_scale() -> None:
    cutoff_1 = estimate_cutoff_from_period(min_period_bars=10, total_bars=100, cutoff_scale=1.0)
    cutoff_2 = estimate_cutoff_from_period(min_period_bars=10, total_bars=100, cutoff_scale=2.0)

    assert cutoff_2 >= cutoff_1


def test_smooth_price_series_basic() -> None:
    prices = np.array([100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0])
    smoothed = smooth_price_series(prices, min_period_bars=2)

    assert len(smoothed) == len(prices)
    assert not np.array_equal(smoothed, prices)


def test_smooth_price_series_empty() -> None:
    prices = np.array([])
    smoothed = smooth_price_series(prices, min_period_bars=2)

    assert len(smoothed) == 0


def test_smooth_price_series_synthetic_trend() -> None:
    t = np.linspace(0, 10, 100)
    trend = 2.0 * t + 100.0
    high_freq_noise = 0.5 * np.sin(20 * t)
    prices = trend + high_freq_noise

    smoothed = smooth_price_series(prices, min_period_bars=20)

    trend_mse = np.mean((smoothed - trend) ** 2)
    noise_mse = np.mean(high_freq_noise**2)

    assert trend_mse < noise_mse


def test_smooth_price_series_cutoff_scale() -> None:
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    trend = 2.0 * t + 100.0
    high_freq = 2.0 * np.sin(30 * t)
    prices = trend + high_freq

    smoothed_1 = smooth_price_series(prices, min_period_bars=10, cutoff_scale=1.0)
    smoothed_2 = smooth_price_series(prices, min_period_bars=10, cutoff_scale=2.0)

    assert not np.array_equal(smoothed_1, smoothed_2)


def test_smooth_price_series_preserves_mean() -> None:
    prices = np.random.randn(100) + 100.0
    smoothed = smooth_price_series(prices, min_period_bars=10)

    assert np.abs(np.mean(smoothed) - np.mean(prices)) < 1.0
