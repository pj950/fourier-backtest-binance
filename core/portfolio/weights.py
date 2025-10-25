import numpy as np
import pandas as pd


def compute_equal_weights(n_assets: int) -> np.ndarray:
    """
    Compute equal weights for all assets.

    Args:
        n_assets: Number of assets in the portfolio

    Returns:
        Array of equal weights summing to 1.0
    """
    if n_assets <= 0:
        return np.array([])
    return np.ones(n_assets) / n_assets


def compute_volatility_scaled_weights(
    returns: pd.DataFrame,
    target_vol: float = 0.02,
    lookback: int = 60,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> np.ndarray:
    """
    Compute volatility-scaled weights (inverse volatility weighting).

    Assets with lower volatility receive higher weights, scaled to target volatility.

    Args:
        returns: DataFrame with returns for each asset (rows=time, cols=assets)
        target_vol: Target portfolio volatility
        lookback: Lookback period for volatility calculation
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset

    Returns:
        Array of weights summing to 1.0
    """
    if returns.empty:
        return np.array([])

    # Calculate rolling volatility using last lookback periods
    recent_returns = returns.tail(lookback)
    volatilities = recent_returns.std(ddof=1)

    # Avoid division by zero
    volatilities = volatilities.replace(0, np.nan)
    volatilities = volatilities.fillna(volatilities.median())
    volatilities = volatilities.clip(lower=1e-8)

    # Inverse volatility weights
    inv_vol = 1.0 / volatilities
    raw_weights = inv_vol / inv_vol.sum()

    # Apply caps
    weights = raw_weights.clip(min_weight, max_weight)
    
    # Normalize to sum to 1.0
    weights = weights / weights.sum()

    return weights.values


def compute_risk_parity_weights(
    returns: pd.DataFrame,
    lookback: int = 60,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> np.ndarray:
    """
    Compute approximate risk parity weights.

    Each asset contributes equally to portfolio risk. This is an iterative
    approximation using marginal risk contributions.

    Args:
        returns: DataFrame with returns for each asset
        lookback: Lookback period for covariance calculation
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset

    Returns:
        Array of risk parity weights summing to 1.0
    """
    if returns.empty:
        return np.array([])

    # Calculate covariance matrix
    recent_returns = returns.tail(lookback)
    cov_matrix = recent_returns.cov().values

    n_assets = len(returns.columns)

    # Start with equal weights
    weights = np.ones(n_assets) / n_assets

    for iteration in range(max_iterations):
        # Calculate portfolio variance
        port_var = weights.T @ cov_matrix @ weights

        if port_var < 1e-10:
            break

        # Calculate marginal risk contributions
        marginal_contrib = cov_matrix @ weights
        risk_contrib = weights * marginal_contrib

        # Target equal risk contribution
        target_rc = port_var / n_assets

        # Update weights proportional to target / actual risk contribution
        risk_contrib_safe = np.clip(risk_contrib, 1e-10, None)
        adjustment = target_rc / risk_contrib_safe
        new_weights = weights * adjustment

        # Apply constraints
        new_weights = np.clip(new_weights, min_weight, max_weight)
        new_weights = new_weights / new_weights.sum()

        # Check convergence
        if np.max(np.abs(new_weights - weights)) < tolerance:
            weights = new_weights
            break

        weights = new_weights

    return weights


def apply_weight_caps(
    weights: np.ndarray,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> np.ndarray:
    """
    Apply minimum and maximum weight constraints and renormalize.

    Args:
        weights: Array of weights
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset

    Returns:
        Capped and normalized weights
    """
    if len(weights) == 0:
        return weights

    # Apply caps
    capped = np.clip(weights, min_weight, max_weight)
    
    # Renormalize
    if capped.sum() > 0:
        capped = capped / capped.sum()
    
    return capped


def compute_market_cap_weights(
    market_caps: dict[str, float],
    symbols: list[str],
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> np.ndarray:
    """
    Compute market-cap weighted portfolio weights.

    Args:
        market_caps: Dictionary mapping symbol to market cap
        symbols: List of symbols in portfolio order
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset

    Returns:
        Array of market-cap weights summing to 1.0
    """
    if not symbols:
        return np.array([])

    caps = np.array([market_caps.get(sym, 1.0) for sym in symbols])
    weights = caps / caps.sum()

    return apply_weight_caps(weights, min_weight, max_weight)


def rebalance_weights(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    threshold: float = 0.05,
) -> tuple[np.ndarray, bool]:
    """
    Determine if rebalancing is needed and return rebalanced weights.

    Rebalancing occurs if any weight deviates from target by more than threshold.

    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        threshold: Rebalancing threshold (absolute deviation)

    Returns:
        Tuple of (new_weights, rebalance_needed)
    """
    if len(current_weights) != len(target_weights):
        return target_weights, True

    max_deviation = np.max(np.abs(current_weights - target_weights))
    rebalance_needed = max_deviation > threshold

    if rebalance_needed:
        return target_weights, True
    else:
        return current_weights, False
