import numpy as np
import pandas as pd


def align_timeframes(
    df_lower: pd.DataFrame,
    df_higher: pd.DataFrame,
    lower_interval: str,
    higher_interval: str,
) -> pd.DataFrame:
    """
    Align lower timeframe data with higher timeframe data.

    For each bar in the lower timeframe, finds the corresponding higher timeframe bar
    by matching timestamps (higher TF bar that contains the lower TF timestamp).

    Args:
        df_lower: Lower timeframe dataframe with 'open_time' column
        df_higher: Higher timeframe dataframe with 'open_time' column
        lower_interval: Lower timeframe interval (e.g., '30m')
        higher_interval: Higher timeframe interval (e.g., '1h', '4h')

    Returns:
        DataFrame with lower TF rows and additional columns from higher TF (prefixed with interval)
    """
    if df_lower.empty or df_higher.empty:
        return df_lower

    df_result = df_lower.copy()

    df_higher_sorted = df_higher.sort_values("open_time").reset_index(drop=True)

    higher_indices = np.searchsorted(
        df_higher_sorted["open_time"].values,
        df_lower["open_time"].values,
        side="right",
    ) - 1

    higher_indices = np.clip(higher_indices, 0, len(df_higher_sorted) - 1)

    suffix = f"_{higher_interval}"
    for col in df_higher_sorted.columns:
        if col != "open_time":
            df_result[col + suffix] = df_higher_sorted.iloc[higher_indices][col].values

    return df_result


def compute_trend_direction(
    close: np.ndarray,
    smoothed: np.ndarray,
    slope_lookback: int = 1,
) -> np.ndarray:
    """
    Compute trend direction: 1 for uptrend, -1 for downtrend, 0 for neutral.

    Uptrend: price > smoothed AND smoothed slope positive
    Downtrend: price < smoothed AND smoothed slope negative
    Neutral: otherwise

    Args:
        close: Close prices
        smoothed: Smoothed trend line
        slope_lookback: Lookback for slope computation

    Returns:
        Array of trend directions {-1, 0, 1}
    """
    if len(close) == 0 or len(smoothed) == 0:
        return np.array([], dtype=int)

    slope = np.full(len(smoothed), 0.0)
    for i in range(slope_lookback, len(smoothed)):
        slope[i] = smoothed[i] - smoothed[i - slope_lookback]

    trend = np.zeros(len(close), dtype=int)

    uptrend = (close > smoothed) & (slope > 0)
    downtrend = (close < smoothed) & (slope < 0)

    trend[uptrend] = 1
    trend[downtrend] = -1

    return trend


def check_mtf_alignment(
    trend_30m: np.ndarray,
    trend_1h: np.ndarray,
    trend_4h: np.ndarray,
    require_all: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Check multi-timeframe trend alignment.

    Args:
        trend_30m: 30m trend direction array (1=up, -1=down, 0=neutral)
        trend_1h: 1h trend direction array (aligned to 30m bars)
        trend_4h: 4h trend direction array (aligned to 30m bars)
        require_all: If True, require all three timeframes to agree.
                     If False, require at least 2 out of 3 to agree.

    Returns:
        Tuple of (aligned_long, aligned_short) boolean arrays
    """
    if len(trend_30m) == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool)

    if require_all:
        aligned_long = (trend_30m == 1) & (trend_1h == 1) & (trend_4h == 1)
        aligned_short = (trend_30m == -1) & (trend_1h == -1) & (trend_4h == -1)
    else:
        long_votes = (trend_30m == 1).astype(int) + (trend_1h == 1).astype(int) + (trend_4h == 1).astype(int)
        short_votes = (trend_30m == -1).astype(int) + (trend_1h == -1).astype(int) + (trend_4h == -1).astype(int)

        aligned_long = long_votes >= 2
        aligned_short = short_votes >= 2

    return aligned_long, aligned_short


def apply_mtf_filter(
    signals: np.ndarray,
    mtf_aligned: np.ndarray,
    direction: int = 1,
) -> np.ndarray:
    """
    Filter entry signals by multi-timeframe alignment.

    Args:
        signals: Entry signals (1=entry, -1=exit, 0=hold)
        mtf_aligned: Boolean array indicating MTF alignment
        direction: 1 for long, -1 for short

    Returns:
        Filtered signals
    """
    if len(signals) == 0:
        return signals

    filtered = signals.copy()

    entry_mask = signals == 1 if direction == 1 else signals == -1

    filtered[entry_mask & ~mtf_aligned] = 0

    return filtered
