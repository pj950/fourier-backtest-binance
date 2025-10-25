import numpy as np
import pandas as pd
import pytest

from core.portfolio.analytics import (
    compute_beta,
    compute_concentration_metrics,
    compute_correlation_matrix,
    compute_diversification_ratio,
    compute_exposure_by_sector,
    compute_portfolio_metrics,
    compute_portfolio_volatility,
    compute_risk_contributions,
    compute_rolling_correlation,
    compute_tracking_error,
)


def test_compute_correlation_matrix():
    """Test correlation matrix computation."""
    np.random.seed(42)
    returns = pd.DataFrame({
        "A": np.random.normal(0, 0.02, 100),
        "B": np.random.normal(0, 0.02, 100),
    })

    corr_matrix = compute_correlation_matrix(returns)

    assert corr_matrix.shape == (2, 2)
    assert np.allclose(np.diag(corr_matrix), 1.0)
    assert corr_matrix.iloc[0, 1] == corr_matrix.iloc[1, 0]


def test_compute_correlation_matrix_empty():
    """Test correlation with empty data."""
    returns = pd.DataFrame()
    corr_matrix = compute_correlation_matrix(returns)
    assert corr_matrix.empty


def test_compute_rolling_correlation():
    """Test rolling correlation computation."""
    np.random.seed(42)
    returns = pd.DataFrame({
        "A": np.random.normal(0, 0.02, 100),
        "B": np.random.normal(0, 0.02, 100),
        "C": np.random.normal(0, 0.02, 100),
    })

    rolling_corr = compute_rolling_correlation(returns, window=30)

    # Should have correlations for each pair
    assert len(rolling_corr) == 3  # (A,B), (A,C), (B,C)
    assert ("A", "B") in rolling_corr
    assert ("A", "C") in rolling_corr
    assert ("B", "C") in rolling_corr

    # Check rolling series length
    for corr_series in rolling_corr.values():
        assert len(corr_series) == 100


def test_compute_portfolio_volatility():
    """Test portfolio volatility calculation."""
    weights = np.array([0.5, 0.5])
    cov_matrix = np.array([[0.01, 0.005], [0.005, 0.01]])

    vol = compute_portfolio_volatility(weights, cov_matrix)

    assert vol > 0
    assert isinstance(vol, float)


def test_compute_portfolio_volatility_zero():
    """Test portfolio volatility with zero covariance."""
    weights = np.array([0.5, 0.5])
    cov_matrix = np.array([[0.0, 0.0], [0.0, 0.0]])

    vol = compute_portfolio_volatility(weights, cov_matrix)

    assert vol == 0.0


def test_compute_diversification_ratio():
    """Test diversification ratio calculation."""
    weights = np.array([0.5, 0.5])
    volatilities = np.array([0.2, 0.2])
    portfolio_vol = 0.15  # Diversification benefit

    div_ratio = compute_diversification_ratio(weights, volatilities, portfolio_vol)

    assert div_ratio > 1.0  # Should be greater than 1 with diversification


def test_compute_diversification_ratio_zero_vol():
    """Test diversification ratio with zero portfolio vol."""
    weights = np.array([0.5, 0.5])
    volatilities = np.array([0.2, 0.2])
    portfolio_vol = 0.0

    div_ratio = compute_diversification_ratio(weights, volatilities, portfolio_vol)

    assert div_ratio == 1.0


def test_compute_concentration_metrics():
    """Test concentration metrics."""
    weights = np.array([0.5, 0.3, 0.2])

    metrics = compute_concentration_metrics(weights)

    assert "herfindahl_index" in metrics
    assert "effective_n" in metrics
    assert "max_weight" in metrics
    assert "top3_concentration" in metrics

    assert metrics["herfindahl_index"] > 0
    assert metrics["effective_n"] > 0
    assert metrics["max_weight"] == 0.5
    assert metrics["top3_concentration"] == 1.0


def test_compute_concentration_metrics_equal_weights():
    """Test concentration with equal weights."""
    weights = np.array([0.25, 0.25, 0.25, 0.25])

    metrics = compute_concentration_metrics(weights)

    # Effective N should be close to 4
    assert np.isclose(metrics["effective_n"], 4.0)
    assert metrics["max_weight"] == 0.25


def test_compute_risk_contributions():
    """Test risk contribution calculation."""
    weights = np.array([0.5, 0.5])
    cov_matrix = np.array([[0.01, 0.005], [0.005, 0.01]])

    risk_contrib = compute_risk_contributions(weights, cov_matrix)

    assert len(risk_contrib) == 2
    assert np.all(risk_contrib > 0)


def test_compute_tracking_error():
    """Test tracking error calculation."""
    np.random.seed(42)
    portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
    benchmark_returns = pd.Series(np.random.normal(0.001, 0.02, 100))

    te = compute_tracking_error(portfolio_returns, benchmark_returns)

    assert te > 0
    assert isinstance(te, float)


def test_compute_tracking_error_empty():
    """Test tracking error with empty data."""
    portfolio_returns = pd.Series([])
    benchmark_returns = pd.Series([])

    te = compute_tracking_error(portfolio_returns, benchmark_returns)

    assert te == 0.0


def test_compute_beta():
    """Test beta calculation."""
    np.random.seed(42)
    market_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
    asset_returns = market_returns * 1.5 + np.random.normal(0, 0.01, 100)

    beta = compute_beta(asset_returns, market_returns)

    # Beta should be around 1.5
    assert beta > 0
    assert 1.0 < beta < 2.0


def test_compute_beta_empty():
    """Test beta with empty data."""
    asset_returns = pd.Series([])
    market_returns = pd.Series([])

    beta = compute_beta(asset_returns, market_returns)

    assert beta == 1.0


def test_compute_portfolio_metrics():
    """Test comprehensive portfolio metrics."""
    np.random.seed(42)
    equity_curve = np.array([10000.0] + list(10000 * np.cumprod(1 + np.random.normal(0.0001, 0.01, 99))))
    weights = np.array([0.5, 0.5])
    returns = pd.DataFrame({
        "A": np.random.normal(0.0001, 0.01, 100),
        "B": np.random.normal(0.0001, 0.01, 100),
    })

    metrics = compute_portfolio_metrics(
        equity_curve=equity_curve,
        weights=weights,
        returns=returns,
        initial_capital=10000.0,
    )

    assert "total_return" in metrics
    assert "annualized_vol" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "max_drawdown_pct" in metrics
    assert "herfindahl_index" in metrics
    assert "effective_n_assets" in metrics
    assert "diversification_ratio" in metrics


def test_compute_exposure_by_sector():
    """Test sector exposure calculation."""
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    symbols = ["BTC", "ETH", "SOL", "AVAX"]
    sector_map = {
        "BTC": "Layer1",
        "ETH": "Layer1",
        "SOL": "Layer1",
        "AVAX": "Layer1",
    }

    exposure = compute_exposure_by_sector(weights, symbols, sector_map)

    assert "Layer1" in exposure
    assert np.isclose(exposure["Layer1"], 1.0)


def test_compute_exposure_by_sector_multiple():
    """Test sector exposure with multiple sectors."""
    weights = np.array([0.4, 0.3, 0.3])
    symbols = ["BTC", "ETH", "UNI"]
    sector_map = {
        "BTC": "Layer1",
        "ETH": "Layer1",
        "UNI": "DeFi",
    }

    exposure = compute_exposure_by_sector(weights, symbols, sector_map)

    assert "Layer1" in exposure
    assert "DeFi" in exposure
    assert np.isclose(exposure["Layer1"], 0.7)
    assert np.isclose(exposure["DeFi"], 0.3)


def test_compute_exposure_unknown_sector():
    """Test sector exposure with unknown sectors."""
    weights = np.array([0.5, 0.5])
    symbols = ["BTC", "XYZ"]
    sector_map = {"BTC": "Layer1"}

    exposure = compute_exposure_by_sector(weights, symbols, sector_map)

    assert "Layer1" in exposure
    assert "Unknown" in exposure
    assert exposure["Layer1"] == 0.5
    assert exposure["Unknown"] == 0.5
