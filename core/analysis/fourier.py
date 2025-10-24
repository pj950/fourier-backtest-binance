import numpy as np
from scipy.fft import dct, idct
from scipy.signal import get_window


def dct_lowpass_smooth(
    signal: np.ndarray,
    cutoff_freq: float,
    taper_width: float = 0.1,
    padding_ratio: float = 0.2,
) -> np.ndarray:
    """
    Apply DCT-based low-pass smoothing with mirrored padding and tapered cutoff.

    Args:
        signal: 1D input signal
        cutoff_freq: Cutoff frequency as fraction of Nyquist (0.0 to 1.0)
        taper_width: Width of the taper region as fraction of cutoff_freq (0.0 to 1.0)
        padding_ratio: Ratio of signal length to use for mirrored padding on each side

    Returns:
        Smoothed signal (same length as input)
    """
    if len(signal) == 0:
        return signal

    n = len(signal)
    pad_len = int(n * padding_ratio)

    left_pad = signal[1 : pad_len + 1][::-1]
    right_pad = signal[-(pad_len + 1) : -1][::-1]
    padded_signal = np.concatenate([left_pad, signal, right_pad])

    dct_coeffs = dct(padded_signal, type=2, norm="ortho")

    cutoff_idx = int(cutoff_freq * len(dct_coeffs))
    taper_len = max(1, int(taper_width * cutoff_idx))

    filter_window = np.ones(len(dct_coeffs))

    if cutoff_idx + taper_len < len(dct_coeffs):
        taper = get_window("hann", 2 * taper_len)
        taper_right = taper[taper_len:]
        filter_window[cutoff_idx : cutoff_idx + taper_len] = taper_right
        filter_window[cutoff_idx + taper_len :] = 0.0
    else:
        filter_window[cutoff_idx:] = 0.0

    filtered_coeffs = dct_coeffs * filter_window

    smoothed_padded = idct(filtered_coeffs, type=2, norm="ortho")

    result: np.ndarray = smoothed_padded[pad_len : pad_len + n]
    return result


def estimate_cutoff_from_period(
    min_period_bars: int,
    total_bars: int,
    cutoff_scale: float = 1.0,
) -> float:
    """
    Estimate cutoff frequency from minimum desired trend period.

    Args:
        min_period_bars: Minimum period to preserve (in bars)
        total_bars: Total number of bars in signal
        cutoff_scale: Scale factor for cutoff (1.0 = at min_period, >1 = more aggressive)

    Returns:
        Cutoff frequency as fraction of Nyquist
    """
    if min_period_bars <= 0:
        return 1.0

    nyquist_freq = total_bars / 2.0
    cutoff_freq_bins = (total_bars / min_period_bars) * cutoff_scale

    return min(cutoff_freq_bins / nyquist_freq, 1.0)


def smooth_price_series(
    prices: np.ndarray,
    min_period_bars: int,
    cutoff_scale: float = 1.0,
    taper_width: float = 0.1,
) -> np.ndarray:
    """
    Smooth a price series using DCT low-pass filter.

    Args:
        prices: Price array (close, high, low, etc.)
        min_period_bars: Minimum trend period to preserve (in bars)
        cutoff_scale: Scale factor for cutoff frequency
        taper_width: Width of taper region

    Returns:
        Smoothed price series
    """
    if len(prices) == 0:
        return prices

    cutoff = estimate_cutoff_from_period(min_period_bars, len(prices), cutoff_scale)

    return dct_lowpass_smooth(prices, cutoff, taper_width)
