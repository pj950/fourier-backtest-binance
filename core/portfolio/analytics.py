import numpy as np
import pandas as pd


def compute_correlation_matrix(
    returns: pd.DataFrame,
    method: str = "pearson",
    min_periods: int = 30,
) -> pd.DataFrame:
    """
    Compute correlation matrix for portfolio assets.

    Args:
        returns: DataFrame with returns for each asset (rows=time, cols=assets)
        method: Correlation method ('pearson', 'spearman', 'kendall')
        min_periods: Minimum number of observations required

    Returns:
        Correlation matrix DataFrame
    """
    if returns.empty:
        return pd.DataFrame()

    return returns.corr(method=method, min_periods=min_periods)


def compute_rolling_correlation(
    returns: pd.DataFrame,
    window: int = 60,
    method: str = "pearson",
) -> dict[tuple[str, str], pd.Series]:
    """
    Compute rolling pairwise correlations between assets.

    Args:
        returns: DataFrame with returns for each asset
        window: Rolling window size
        method: Correlation method

    Returns:
        Dictionary mapping (asset1, asset2) pairs to correlation Series
    """
    if returns.empty or len(returns.columns) < 2:
        return {}

    correlations = {}
    symbols = returns.columns.tolist()

    for i, sym1 in enumerate(symbols):
        for sym2 in symbols[i + 1:]:
            corr = returns[sym1].rolling(window).corr(returns[sym2])
            correlations[(sym1, sym2)] = corr

    return correlations


def compute_portfolio_volatility(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """
    Compute portfolio volatility from weights and covariance matrix.

    Args:
        weights: Portfolio weights
        cov_matrix: Covariance matrix of asset returns

    Returns:
        Portfolio volatility (standard deviation)
    """
    if len(weights) == 0 or len(cov_matrix) == 0:
        return 0.0

    portfolio_var = weights.T @ cov_matrix @ weights
    return np.sqrt(max(portfolio_var, 0.0))


def compute_diversification_ratio(
    weights: np.ndarray,
    volatilities: np.ndarray,
    portfolio_vol: float,
) -> float:
    """
    Compute diversification ratio.

    Ratio of weighted average volatility to portfolio volatility.
    Higher values indicate better diversification.

    Args:
        weights: Portfolio weights
        volatilities: Individual asset volatilities
        portfolio_vol: Portfolio volatility

    Returns:
        Diversification ratio
    """
    if portfolio_vol < 1e-10 or len(weights) == 0:
        return 1.0

    weighted_avg_vol = np.sum(weights * volatilities)
    return weighted_avg_vol / portfolio_vol


def compute_concentration_metrics(weights: np.ndarray) -> dict[str, float]:
    """
    Compute portfolio concentration metrics.

    Args:
        weights: Portfolio weights

    Returns:
        Dictionary with concentration metrics:
        - herfindahl_index: Sum of squared weights
        - effective_n: Effective number of assets (1 / HHI)
        - max_weight: Maximum single asset weight
        - top3_concentration: Sum of top 3 weights
    """
    if len(weights) == 0:
        return {
            "herfindahl_index": 0.0,
            "effective_n": 0.0,
            "max_weight": 0.0,
            "top3_concentration": 0.0,
        }

    hhi = np.sum(weights ** 2)
    effective_n = 1.0 / hhi if hhi > 0 else 0.0
    max_weight = np.max(weights)
    
    sorted_weights = np.sort(weights)[::-1]
    top3_concentration = np.sum(sorted_weights[:min(3, len(sorted_weights))])

    return {
        "herfindahl_index": hhi,
        "effective_n": effective_n,
        "max_weight": max_weight,
        "top3_concentration": top3_concentration,
    }


def compute_risk_contributions(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute risk contribution of each asset to portfolio volatility.

    Args:
        weights: Portfolio weights
        cov_matrix: Covariance matrix

    Returns:
        Array of risk contributions (sum to portfolio variance)
    """
    if len(weights) == 0 or len(cov_matrix) == 0:
        return np.array([])

    marginal_contrib = cov_matrix @ weights
    risk_contrib = weights * marginal_contrib

    return risk_contrib


def compute_tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Compute tracking error relative to a benchmark.

    Args:
        portfolio_returns: Portfolio returns Series
        benchmark_returns: Benchmark returns Series

    Returns:
        Tracking error (annualized)
    """
    if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0

    # Align series
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner")
    if aligned.empty:
        return 0.0

    diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = diff.std() * np.sqrt(252)  # Annualized

    return te


def compute_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
) -> float:
    """
    Compute beta of an asset relative to market.

    Args:
        asset_returns: Asset returns Series
        market_returns: Market returns Series

    Returns:
        Beta coefficient
    """
    if len(asset_returns) == 0 or len(market_returns) == 0:
        return 1.0

    # Align series
    aligned = pd.concat([asset_returns, market_returns], axis=1, join="inner")
    if len(aligned) < 2:
        return 1.0

    covariance = aligned.cov().iloc[0, 1]
    market_variance = aligned.iloc[:, 1].var()

    if market_variance < 1e-10:
        return 1.0

    return covariance / market_variance


def compute_portfolio_metrics(
    equity_curve: np.ndarray,
    weights: np.ndarray,
    returns: pd.DataFrame,
    initial_capital: float,
) -> dict[str, float]:
    """
    Compute comprehensive portfolio-level metrics.

    Args:
        equity_curve: Portfolio equity over time
        weights: Current portfolio weights
        returns: DataFrame of asset returns
        initial_capital: Starting capital

    Returns:
        Dictionary of portfolio metrics
    """
    if len(equity_curve) == 0:
        return {}

    # Basic return metrics
    total_return = (equity_curve[-1] - initial_capital) / initial_capital
    
    port_returns = np.diff(equity_curve) / equity_curve[:-1]
    port_returns = port_returns[~np.isnan(port_returns)]

    # Risk metrics
    port_vol = np.std(port_returns, ddof=1) * np.sqrt(252 * 24)  # Annualized hourly
    
    # Sharpe ratio
    if port_vol > 1e-10:
        sharpe = (np.mean(port_returns) / np.std(port_returns, ddof=1)) * np.sqrt(252 * 24)
    else:
        sharpe = 0.0

    # Drawdown
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - cummax
    max_dd = np.min(drawdown)
    max_dd_pct = max_dd / cummax[np.argmin(drawdown)] if cummax[np.argmin(drawdown)] > 0 else 0.0

    # Concentration
    concentration = compute_concentration_metrics(weights)

    # Diversification
    if not returns.empty and len(returns.columns) > 0:
        cov_matrix = returns.cov().values
        asset_vols = returns.std().values
        portfolio_vol_annual = compute_portfolio_volatility(weights, cov_matrix) * np.sqrt(252 * 24)
        div_ratio = compute_diversification_ratio(weights, asset_vols * np.sqrt(252 * 24), portfolio_vol_annual)
    else:
        div_ratio = 1.0

    return {
        "total_return": total_return,
        "annualized_vol": port_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "herfindahl_index": concentration["herfindahl_index"],
        "effective_n_assets": concentration["effective_n"],
        "max_weight": concentration["max_weight"],
        "diversification_ratio": div_ratio,
    }


def compute_exposure_by_sector(
    weights: np.ndarray,
    symbols: list[str],
    sector_map: dict[str, str],
) -> dict[str, float]:
    """
    Compute portfolio exposure by sector.

    Args:
        weights: Portfolio weights
        symbols: List of symbols
        sector_map: Dictionary mapping symbol to sector

    Returns:
        Dictionary mapping sector to total weight
    """
    if len(weights) != len(symbols):
        return {}

    sector_exposure: dict[str, float] = {}

    for symbol, weight in zip(symbols, weights):
        sector = sector_map.get(symbol, "Unknown")
        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + weight

    return sector_exposure
