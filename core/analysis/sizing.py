import numpy as np


def compute_volatility_based_size(
    capital: float,
    entry_price: float,
    stop_price: float,
    volatility: float,
    risk_target: float,
    max_risk_per_trade: float,
    vol_target_mode: str = "atr",
) -> float:
    """
    Compute position size based on volatility targeting.

    Args:
        capital: Available capital
        entry_price: Entry price
        stop_price: Stop loss price
        volatility: Current volatility measure (ATR or sigma)
        risk_target: Target volatility/risk as fraction of capital
        max_risk_per_trade: Maximum risk per trade as fraction of capital
        vol_target_mode: 'atr' or 'sigma' for volatility targeting method

    Returns:
        Position size (in base currency units)
    """
    if capital <= 0 or entry_price <= 0 or volatility <= 0:
        return 0.0

    risk_per_unit = abs(entry_price - stop_price)
    if risk_per_unit <= 0:
        risk_per_unit = volatility

    if vol_target_mode == "atr":
        target_risk = capital * risk_target
        size = target_risk / risk_per_unit
    elif vol_target_mode == "sigma":
        target_risk = capital * risk_target
        size = target_risk / volatility
    else:
        size = (capital * risk_target) / risk_per_unit

    max_risk_amount = capital * max_risk_per_trade
    max_size = max_risk_amount / risk_per_unit if risk_per_unit > 0 else 0

    size = min(size, max_size)

    max_size_by_capital = capital / entry_price
    size = min(size, max_size_by_capital)

    return max(size, 0.0)


def compute_fixed_risk_size(
    capital: float,
    entry_price: float,
    stop_price: float,
    risk_fraction: float,
) -> float:
    """
    Compute position size based on fixed risk per trade.

    Args:
        capital: Available capital
        entry_price: Entry price
        stop_price: Stop loss price
        risk_fraction: Risk as fraction of capital (e.g., 0.01 for 1%)

    Returns:
        Position size (in base currency units)
    """
    if capital <= 0 or entry_price <= 0:
        return 0.0

    risk_per_unit = abs(entry_price - stop_price)
    if risk_per_unit <= 0:
        return 0.0

    risk_amount = capital * risk_fraction
    size = risk_amount / risk_per_unit

    max_size_by_capital = capital / entry_price
    size = min(size, max_size_by_capital)

    return max(size, 0.0)


def compute_pyramid_size(
    initial_size: float,
    current_position: float,
    max_pyramids: int,
    pyramid_scale: float = 0.5,
) -> float:
    """
    Compute additional position size for pyramiding.

    Args:
        initial_size: Initial position size
        current_position: Current total position size
        max_pyramids: Maximum number of pyramid additions allowed
        pyramid_scale: Scale factor for each pyramid (e.g., 0.5 means each add is 50% of previous)

    Returns:
        Additional size for pyramid, or 0 if max pyramids reached
    """
    if initial_size <= 0 or current_position < initial_size:
        return 0.0

    ratio = current_position / initial_size
    num_pyramids = int(np.log(ratio) / np.log(1.0 + pyramid_scale)) + 1

    if num_pyramids >= max_pyramids:
        return 0.0

    additional_size = initial_size * pyramid_scale
    return additional_size


def check_pyramid_conditions(
    entry_price: float,
    current_price: float,
    profit_threshold: float,
    direction: int,
) -> bool:
    """
    Check if conditions are met for pyramiding into position.

    Args:
        entry_price: Original entry price
        current_price: Current price
        profit_threshold: Minimum profit % required to pyramid (e.g., 0.02 for 2%)
        direction: 1 for long, -1 for short

    Returns:
        True if pyramiding conditions met
    """
    if entry_price <= 0:
        return False

    profit_pct = (current_price - entry_price) / entry_price

    if direction == 1:
        return profit_pct >= profit_threshold
    else:
        return profit_pct <= -profit_threshold
