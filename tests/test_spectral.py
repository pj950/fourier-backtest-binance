import numpy as np

from core.analysis.spectral import (
    compute_fft_spectrum,
    compute_sliding_dominant_period,
    compute_welch_psd,
    find_dominant_peaks,
    format_period,
)


def test_compute_fft_spectrum_basic() -> None:
    signal = np.sin(2 * np.pi * 0.1 * np.arange(100))
    frequencies, power_spectrum = compute_fft_spectrum(signal)

    assert len(frequencies) > 0
    assert len(power_spectrum) == len(frequencies)
    assert np.all(frequencies > 0)
    assert np.all(power_spectrum >= 0)


def test_compute_fft_spectrum_empty() -> None:
    signal = np.array([])
    frequencies, power_spectrum = compute_fft_spectrum(signal)

    assert len(frequencies) == 0
    assert len(power_spectrum) == 0


def test_compute_fft_spectrum_single_frequency() -> None:
    freq = 0.1
    signal = np.sin(2 * np.pi * freq * np.arange(200))
    frequencies, power_spectrum = compute_fft_spectrum(signal)

    peak_idx = np.argmax(power_spectrum)
    detected_freq = frequencies[peak_idx]

    assert np.abs(detected_freq - freq) < 0.01


def test_find_dominant_peaks_basic() -> None:
    frequencies = np.linspace(0.01, 0.5, 100)
    power_spectrum = np.random.rand(100)
    power_spectrum[20] = 10.0
    power_spectrum[50] = 8.0
    power_spectrum[80] = 6.0

    peaks = find_dominant_peaks(frequencies, power_spectrum, n_peaks=3)

    assert len(peaks) <= 3
    assert all("frequency" in p for p in peaks)
    assert all("power" in p for p in peaks)
    assert all("period" in p for p in peaks)


def test_find_dominant_peaks_empty() -> None:
    frequencies = np.array([])
    power_spectrum = np.array([])

    peaks = find_dominant_peaks(frequencies, power_spectrum)

    assert len(peaks) == 0


def test_find_dominant_peaks_sorted_by_power() -> None:
    frequencies = np.linspace(0.01, 0.5, 100)
    power_spectrum = np.random.rand(100)
    power_spectrum[20] = 10.0
    power_spectrum[50] = 15.0
    power_spectrum[80] = 5.0

    peaks = find_dominant_peaks(frequencies, power_spectrum, n_peaks=3)

    assert peaks[0]["power"] >= peaks[1]["power"]
    assert peaks[1]["power"] >= peaks[2]["power"]


def test_find_dominant_peaks_min_power_ratio() -> None:
    frequencies = np.linspace(0.01, 0.5, 100)
    power_spectrum = np.ones(100)
    power_spectrum[50] = 100.0

    peaks = find_dominant_peaks(frequencies, power_spectrum, n_peaks=10, min_power_ratio=0.5)

    assert len(peaks) <= 1


def test_compute_welch_psd_basic() -> None:
    signal = np.random.randn(500)
    frequencies, psd = compute_welch_psd(signal, window_length=128, overlap_ratio=0.5)

    assert len(frequencies) > 0
    assert len(psd) == len(frequencies)
    assert np.all(psd >= 0)


def test_compute_welch_psd_short_signal() -> None:
    signal = np.random.randn(50)
    frequencies, psd = compute_welch_psd(signal, window_length=256, overlap_ratio=0.5)

    assert len(frequencies) > 0
    assert len(psd) == len(frequencies)


def test_compute_welch_psd_overlap() -> None:
    signal = np.random.randn(500)

    freqs_1, psd_1 = compute_welch_psd(signal, window_length=128, overlap_ratio=0.0)
    freqs_2, psd_2 = compute_welch_psd(signal, window_length=128, overlap_ratio=0.5)

    assert len(freqs_1) == len(freqs_2)
    assert not np.array_equal(psd_1, psd_2)


def test_compute_sliding_dominant_period_basic() -> None:
    signal = np.sin(2 * np.pi * 0.1 * np.arange(500))
    time_indices, dominant_periods = compute_sliding_dominant_period(
        signal, window_length=128, overlap_ratio=0.5
    )

    assert len(time_indices) > 0
    assert len(dominant_periods) == len(time_indices)


def test_compute_sliding_dominant_period_short_signal() -> None:
    signal = np.random.randn(50)
    time_indices, dominant_periods = compute_sliding_dominant_period(
        signal, window_length=256, overlap_ratio=0.5
    )

    assert len(time_indices) > 0


def test_compute_sliding_dominant_period_detects_period() -> None:
    period = 10
    signal = np.sin(2 * np.pi * (1 / period) * np.arange(500))
    time_indices, dominant_periods = compute_sliding_dominant_period(
        signal, window_length=256, overlap_ratio=0.5
    )

    valid_periods = dominant_periods[~np.isnan(dominant_periods)]

    if len(valid_periods) > 0:
        mean_period = np.mean(valid_periods)
        assert np.abs(mean_period - period) < period * 0.3


def test_format_period_hours() -> None:
    period_str = format_period(12.0, "1h")
    assert "12.0 bars" in period_str
    assert "12.0h" in period_str


def test_format_period_days() -> None:
    period_str = format_period(48.0, "1h")
    assert "48.0 bars" in period_str
    assert "2.0d" in period_str


def test_format_period_30m() -> None:
    period_str = format_period(48.0, "30m")
    assert "48.0 bars" in period_str
    assert "24.0h" in period_str or "1.0d" in period_str


def test_format_period_4h() -> None:
    period_str = format_period(6.0, "4h")
    assert "6.0 bars" in period_str
    assert "24.0h" in period_str or "1.0d" in period_str
