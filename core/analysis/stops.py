import numpy as np
import pandas as pd


def compute_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    Compute Average True Range (ATR).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period

    Returns:
        ATR values
    """
    if len(high) == 0:
        return np.array([])

    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]

    for i in range(1, len(high)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    return atr


def compute_residual_sigma(
    prices: np.ndarray,
    smoothed: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """
    Compute rolling standard deviation of residuals (price - smoothed).

    Args:
        prices: Raw prices
        smoothed: Smoothed trend line
        window: Rolling window for standard deviation

    Returns:
        Rolling standard deviation of residuals
    """
    if len(prices) == 0:
        return np.array([])

    residuals = prices - smoothed
    sigma = pd.Series(residuals).rolling(window=window, min_periods=1).std().values
    sigma = np.where(np.isnan(sigma), 0.0, sigma)
    return sigma


def compute_residual_quantile(
    prices: np.ndarray,
    smoothed: np.ndarray,
    window: int = 20,
    quantile: float = 0.95,
) -> np.ndarray:
    """
    Compute rolling quantile of absolute residuals.

    Args:
        prices: Raw prices
        smoothed: Smoothed trend line
        window: Rolling window for quantile computation
        quantile: Quantile level (0-1)

    Returns:
        Rolling quantile of absolute residuals
    """
    if len(prices) == 0:
        return np.array([])

    residuals = np.abs(prices - smoothed)
    quant = (
        pd.Series(residuals).rolling(window=window, min_periods=1).quantile(quantile).values
    )
    quant = np.where(np.isnan(quant), 0.0, quant)
    return quant


def compute_atr_stops(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr_period: int = 14,
    k_stop: float = 2.0,
    k_profit: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ATR-based stop and take-profit bands.

    Args:
        close: Close prices
        high: High prices
        low: Low prices
        atr_period: Period for ATR calculation
        k_stop: Multiplier for stop loss (distance in ATR units)
        k_profit: Multiplier for take profit (distance in ATR units)

    Returns:
        Tuple of (long_stop, long_profit, short_stop, short_profit)
    """
    atr = compute_atr(high, low, close, period=atr_period)

    long_stop = close - k_stop * atr
    long_profit = close + k_profit * atr
    short_stop = close + k_stop * atr
    short_profit = close - k_profit * atr

    return long_stop, long_profit, short_stop, short_profit


def compute_residual_stops(
    close: np.ndarray,
    smoothed: np.ndarray,
    method: str = "sigma",
    window: int = 20,
    quantile: float = 0.95,
    k_stop: float = 2.0,
    k_profit: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute residual-based stop and take-profit bands.

    Args:
        close: Close prices
        smoothed: Smoothed trend line
        method: 'sigma' for standard deviation or 'quantile' for quantile-based
        window: Rolling window for computation
        quantile: Quantile level (only used if method='quantile')
        k_stop: Multiplier for stop loss
        k_profit: Multiplier for take profit

    Returns:
        Tuple of (long_stop, long_profit, short_stop, short_profit)
    """
    if method == "sigma":
        bandwidth = compute_residual_sigma(close, smoothed, window=window)
    elif method == "quantile":
        bandwidth = compute_residual_quantile(close, smoothed, window=window, quantile=quantile)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sigma' or 'quantile'.")

    long_stop = smoothed - k_stop * bandwidth
    long_profit = smoothed + k_profit * bandwidth
    short_stop = smoothed + k_stop * bandwidth
    short_profit = smoothed - k_profit * bandwidth

    return long_stop, long_profit, short_stop, short_profit


def compute_trailing_stop(
    prices: np.ndarray,
    entry_price: float,
    entry_idx: int,
    direction: int,
    stop_levels: np.ndarray,
) -> np.ndarray:
    """
    Compute trailing stop that moves favorably but never reverses.

    Args:
        prices: Price array
        entry_price: Entry price
        entry_idx: Entry bar index
        direction: 1 for long, -1 for short
        stop_levels: Initial stop levels for each bar

    Returns:
        Trailing stop levels (only valid from entry_idx onwards)
    """
    if len(prices) == 0 or entry_idx >= len(prices):
        return np.array([])

    trailing = np.full(len(prices), np.nan)
    trailing[entry_idx] = stop_levels[entry_idx]

    if direction == 1:
        for i in range(entry_idx + 1, len(prices)):
            trailing[i] = max(trailing[i - 1], stop_levels[i])
    else:
        for i in range(entry_idx + 1, len(prices)):
            trailing[i] = min(trailing[i - 1], stop_levels[i])

    return trailing
