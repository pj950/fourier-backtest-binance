import numpy as np


def compute_slope(signal: np.ndarray, lookback: int = 1) -> np.ndarray:
    """
    Compute slope of a signal using simple difference.

    Args:
        signal: Input signal
        lookback: Lookback period for slope computation

    Returns:
        Slope values (NaN for first lookback bars)
    """
    if len(signal) == 0:
        return np.array([])

    slope = np.full(len(signal), np.nan)
    for i in range(lookback, len(signal)):
        slope[i] = signal[i] - signal[i - lookback]

    return slope


def detect_cross_above(
    signal: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Detect when signal crosses above reference.

    Args:
        signal: Primary signal
        reference: Reference signal

    Returns:
        Boolean array: True where cross above occurs
    """
    if len(signal) == 0 or len(reference) == 0:
        return np.array([], dtype=bool)

    below = signal[:-1] <= reference[:-1]
    above = signal[1:] > reference[1:]
    cross_above = np.zeros(len(signal), dtype=bool)
    cross_above[1:] = below & above

    return cross_above


def detect_cross_below(
    signal: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Detect when signal crosses below reference.

    Args:
        signal: Primary signal
        reference: Reference signal

    Returns:
        Boolean array: True where cross below occurs
    """
    if len(signal) == 0 or len(reference) == 0:
        return np.array([], dtype=bool)

    above = signal[:-1] >= reference[:-1]
    below = signal[1:] < reference[1:]
    cross_below = np.zeros(len(signal), dtype=bool)
    cross_below[1:] = above & below

    return cross_below


def generate_trend_following_signals(
    close: np.ndarray,
    smoothed: np.ndarray,
    slope_threshold: float = 0.0,
    slope_lookback: int = 1,
    min_volatility: float = 0.0,
    volatility: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate trend-following entry and exit signals.

    Entry conditions:
    - Price crosses above smoothed trend
    - Trend has positive slope
    - Volatility above minimum (if specified)

    Exit conditions:
    - Price crosses below smoothed trend

    Args:
        close: Close prices
        smoothed: Smoothed trend line
        slope_threshold: Minimum slope for entry (default 0 = positive slope required)
        slope_lookback: Lookback period for slope computation
        min_volatility: Minimum volatility filter (optional)
        volatility: Volatility measure (e.g., ATR or residual sigma), optional

    Returns:
        Tuple of (entry_signals, exit_signals) as boolean arrays
    """
    if len(close) == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool)

    slope = compute_slope(smoothed, lookback=slope_lookback)

    cross_above = detect_cross_above(close, smoothed)
    cross_below = detect_cross_below(close, smoothed)

    positive_slope = slope > slope_threshold

    entry = cross_above & positive_slope

    if min_volatility > 0.0 and volatility is not None:
        vol_filter = volatility >= min_volatility
        entry = entry & vol_filter

    exit_signal = cross_below

    return entry, exit_signal


def generate_signals_with_stops(
    close: np.ndarray,
    smoothed: np.ndarray,
    stop_levels: np.ndarray,
    slope_threshold: float = 0.0,
    slope_lookback: int = 1,
    min_volatility: float = 0.0,
    volatility: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generate position signals (-1, 0, 1) incorporating stop loss logic.

    Signal values:
    - 1: Long entry
    - 0: Neutral/flat
    - -1: Exit (from long)

    Entry conditions same as generate_trend_following_signals.
    Exit on cross below trend OR stop hit.

    Args:
        close: Close prices
        smoothed: Smoothed trend line
        stop_levels: Stop loss levels (for long positions)
        slope_threshold: Minimum slope for entry
        slope_lookback: Lookback period for slope
        min_volatility: Minimum volatility filter
        volatility: Volatility measure, optional

    Returns:
        Signal array with values {-1, 0, 1}
    """
    entry, trend_exit = generate_trend_following_signals(
        close=close,
        smoothed=smoothed,
        slope_threshold=slope_threshold,
        slope_lookback=slope_lookback,
        min_volatility=min_volatility,
        volatility=volatility,
    )

    stop_exit = close < stop_levels

    exit_signal = trend_exit | stop_exit

    signals = np.zeros(len(close), dtype=int)
    signals[entry] = 1
    signals[exit_signal] = -1

    return signals


def filter_signals_by_period(
    signals: np.ndarray,
    min_bars_between: int,
) -> np.ndarray:
    """
    Filter signals to enforce minimum bars between entries.

    Args:
        signals: Signal array (1=entry, -1=exit, 0=hold)
        min_bars_between: Minimum bars required between consecutive entries

    Returns:
        Filtered signal array
    """
    if len(signals) == 0:
        return signals

    filtered = signals.copy()
    last_entry_idx = -min_bars_between

    for i in range(len(signals)):
        if signals[i] == 1:
            if i - last_entry_idx < min_bars_between:
                filtered[i] = 0
            else:
                last_entry_idx = i

    return filtered
