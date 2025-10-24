import numpy as np


def check_time_based_exit(
    entry_idx: int,
    current_idx: int,
    max_bars_held: int,
) -> bool:
    """
    Check if position should exit based on time held.

    Args:
        entry_idx: Bar index when position was entered
        current_idx: Current bar index
        max_bars_held: Maximum bars to hold position

    Returns:
        True if time-based exit triggered
    """
    if entry_idx < 0:
        return False

    bars_held = current_idx - entry_idx
    return bars_held >= max_bars_held


def compute_partial_tp_levels(
    entry_price: float,
    direction: int,
    scales: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Compute partial take-profit levels.

    Args:
        entry_price: Entry price
        direction: 1 for long, -1 for short
        scales: List of (price_pct, size_pct) tuples
                price_pct: price movement from entry (e.g., 0.02 for 2%)
                size_pct: fraction of position to close (e.g., 0.5 for 50%)

    Returns:
        List of (price_level, size_fraction) tuples
    """
    levels = []
    for price_pct, size_pct in scales:
        if direction == 1:
            price_level = entry_price * (1 + price_pct)
        else:
            price_level = entry_price * (1 - price_pct)

        levels.append((price_level, size_pct))

    return levels


def check_partial_tp_hit(
    current_price: float,
    high_price: float,
    low_price: float,
    tp_levels: list[tuple[float, float]],
    hit_levels: set[int],
    direction: int,
) -> list[int]:
    """
    Check which partial take-profit levels were hit in current bar.

    Args:
        current_price: Current close price
        high_price: High price of current bar
        low_price: Low price of current bar
        tp_levels: List of (price_level, size_fraction) tuples
        hit_levels: Set of already hit level indices
        direction: 1 for long, -1 for short

    Returns:
        List of newly hit level indices
    """
    newly_hit = []

    for i, (price_level, _) in enumerate(tp_levels):
        if i in hit_levels:
            continue

        if direction == 1:
            if high_price >= price_level:
                newly_hit.append(i)
        else:
            if low_price <= price_level:
                newly_hit.append(i)

    return newly_hit


def compute_slope_reversal(
    smoothed: np.ndarray,
    lookback: int = 2,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Detect slope reversal in smoothed trend.

    Args:
        smoothed: Smoothed price array
        lookback: Lookback for slope computation
        threshold: Minimum absolute slope change to consider reversal

    Returns:
        Boolean array: True where reversal detected
    """
    if len(smoothed) < lookback + 1:
        return np.zeros(len(smoothed), dtype=bool)

    slope = np.full(len(smoothed), 0.0)
    for i in range(lookback, len(smoothed)):
        slope[i] = smoothed[i] - smoothed[i - lookback]

    reversal = np.zeros(len(smoothed), dtype=bool)

    for i in range(1, len(slope)):
        prev_slope = slope[i - 1]
        curr_slope = slope[i]

        if abs(prev_slope) > threshold and abs(curr_slope) > threshold:
            if (prev_slope > 0 and curr_slope < 0) or (prev_slope < 0 and curr_slope > 0):
                reversal[i] = True

    return reversal


def combine_exit_conditions(
    *conditions: np.ndarray,
) -> np.ndarray:
    """
    Combine multiple exit condition arrays with logical OR.

    Args:
        *conditions: Variable number of boolean arrays

    Returns:
        Combined boolean array (True if any condition is True)
    """
    if not conditions:
        return np.array([], dtype=bool)

    result = conditions[0].copy()
    for condition in conditions[1:]:
        result = result | condition

    return result
