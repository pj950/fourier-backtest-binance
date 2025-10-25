import numpy as np
import pandas as pd
import pytest

from core.backtest.engine import BacktestConfig
from core.portfolio.executor import (
    SymbolBacktestResult,
    align_symbol_results,
    compute_symbol_statistics,
    run_single_symbol_backtest,
)
from core.portfolio.portfolio import Portfolio, PortfolioConfig, create_portfolio


def create_test_dataframe(n_bars: int = 100, base_price: float = 100.0) -> pd.DataFrame:
    """Create a test OHLCV dataframe."""
    np.random.seed(42)
    close = base_price + np.cumsum(np.random.normal(0, 1, n_bars))
    high = close + np.abs(np.random.normal(0, 0.5, n_bars))
    low = close - np.abs(np.random.normal(0, 0.5, n_bars))
    open_prices = close + np.random.normal(0, 0.3, n_bars)

    df = pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n_bars, freq="1h"),
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.uniform(1000, 10000, n_bars),
    })

    return df


def simple_strategy(df: pd.DataFrame, **params) -> np.ndarray:
    """Simple moving average crossover strategy."""
    close = df["close"].values
    n = len(close)
    signals = np.zeros(n)

    # Simple: buy when price crosses above mean, sell when below
    mean_price = np.mean(close[:20])

    in_position = False
    for i in range(20, n):
        if not in_position and close[i] > mean_price:
            signals[i] = 1
            in_position = True
        elif in_position and close[i] < mean_price:
            signals[i] = -1
            in_position = False

    return signals


def test_create_portfolio():
    """Test portfolio creation with defaults."""
    symbols = ["BTC", "ETH", "SOL"]
    portfolio = create_portfolio(symbols, weighting_method="equal")

    assert portfolio.symbols == symbols
    assert portfolio.portfolio_config.weighting_method == "equal"
    assert len(portfolio.weights) == 3
    assert np.isclose(portfolio.weights.sum(), 1.0)


def test_portfolio_compute_weights_equal():
    """Test equal weight computation."""
    symbols = ["BTC", "ETH"]
    portfolio = create_portfolio(symbols, weighting_method="equal")

    np.random.seed(42)
    returns_df = pd.DataFrame({
        "BTC": np.random.normal(0, 0.02, 100),
        "ETH": np.random.normal(0, 0.02, 100),
    })

    weights = portfolio.compute_weights(returns_df, method="equal")

    assert len(weights) == 2
    assert np.allclose(weights, 0.5)


def test_portfolio_compute_weights_volatility():
    """Test volatility-based weight computation."""
    symbols = ["BTC", "ETH"]
    portfolio = create_portfolio(symbols, weighting_method="volatility")

    np.random.seed(42)
    returns_df = pd.DataFrame({
        "BTC": np.random.normal(0, 0.01, 100),  # Lower vol
        "ETH": np.random.normal(0, 0.03, 100),  # Higher vol
    })

    weights = portfolio.compute_weights(returns_df, method="volatility")

    assert len(weights) == 2
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > weights[1]  # BTC should have higher weight


def test_run_single_symbol_backtest():
    """Test single symbol backtest execution."""
    df = create_test_dataframe(n_bars=100)
    config = BacktestConfig(initial_capital=10000.0)

    result = run_single_symbol_backtest(
        symbol="BTC",
        df=df,
        strategy_func=simple_strategy,
        strategy_params={},
        config=config,
    )

    assert result.symbol == "BTC"
    assert len(result.equity_curve) == 100
    assert len(result.returns) == 100
    assert result.result.metrics is not None


def test_align_symbol_results():
    """Test alignment of symbol results."""
    # Create test results
    df1 = create_test_dataframe(n_bars=100, base_price=100.0)
    df2 = create_test_dataframe(n_bars=100, base_price=200.0)

    config = BacktestConfig(initial_capital=10000.0)

    result1 = run_single_symbol_backtest("BTC", df1, simple_strategy, {}, config)
    result2 = run_single_symbol_backtest("ETH", df2, simple_strategy, {}, config)

    results = {"BTC": result1, "ETH": result2}

    equity_df, returns_df = align_symbol_results(results)

    assert equity_df.shape == (100, 2)
    assert returns_df.shape == (100, 2)
    assert "BTC" in equity_df.columns
    assert "ETH" in equity_df.columns


def test_compute_symbol_statistics():
    """Test computation of symbol statistics."""
    df1 = create_test_dataframe(n_bars=100, base_price=100.0)
    df2 = create_test_dataframe(n_bars=100, base_price=200.0)

    config = BacktestConfig(initial_capital=10000.0)

    result1 = run_single_symbol_backtest("BTC", df1, simple_strategy, {}, config)
    result2 = run_single_symbol_backtest("ETH", df2, simple_strategy, {}, config)

    results = {"BTC": result1, "ETH": result2}

    stats = compute_symbol_statistics(results)

    assert len(stats) == 2
    assert "symbol" in stats.columns
    assert "total_return" in stats.columns
    assert "sharpe_ratio" in stats.columns
    assert "n_trades" in stats.columns


def test_portfolio_backtest_equal_weights():
    """Test full portfolio backtest with equal weights."""
    symbols = ["BTC", "ETH"]
    
    # Create test data
    data_dict = {
        "BTC": create_test_dataframe(n_bars=100, base_price=100.0),
        "ETH": create_test_dataframe(n_bars=100, base_price=200.0),
    }

    portfolio_config = PortfolioConfig(
        weighting_method="equal",
        rebalance_frequency=24,
    )

    backtest_config = BacktestConfig(initial_capital=10000.0)

    portfolio = Portfolio(
        symbols=symbols,
        portfolio_config=portfolio_config,
        backtest_config=backtest_config,
    )

    result = portfolio.run_backtest(
        data_dict=data_dict,
        strategy_func=simple_strategy,
        strategy_params={},
    )

    assert len(result.equity_curve) == 100
    assert len(result.weights) == 2
    assert np.isclose(result.weights.sum(), 1.0)
    assert result.symbols == symbols
    assert "BTC" in result.symbol_results
    assert "ETH" in result.symbol_results
    assert not result.correlation_matrix.empty
    assert "total_return" in result.metrics
    assert "sharpe_ratio" in result.metrics


def test_portfolio_backtest_volatility_weights():
    """Test portfolio backtest with volatility weighting."""
    symbols = ["BTC", "ETH"]
    
    # Create test data with different volatilities
    np.random.seed(42)
    data_dict = {
        "BTC": create_test_dataframe(n_bars=100, base_price=100.0),
        "ETH": create_test_dataframe(n_bars=100, base_price=200.0),
    }

    portfolio_config = PortfolioConfig(
        weighting_method="volatility",
        rebalance_frequency=24,
        lookback_period=30,
    )

    backtest_config = BacktestConfig(initial_capital=10000.0)

    portfolio = Portfolio(
        symbols=symbols,
        portfolio_config=portfolio_config,
        backtest_config=backtest_config,
    )

    result = portfolio.run_backtest(
        data_dict=data_dict,
        strategy_func=simple_strategy,
        strategy_params={},
    )

    assert len(result.equity_curve) == 100
    assert len(result.weights) == 2
    assert np.isclose(result.weights.sum(), 1.0)


def test_portfolio_backtest_rebalancing():
    """Test portfolio rebalancing behavior."""
    symbols = ["BTC", "ETH"]
    
    data_dict = {
        "BTC": create_test_dataframe(n_bars=100, base_price=100.0),
        "ETH": create_test_dataframe(n_bars=100, base_price=200.0),
    }

    portfolio_config = PortfolioConfig(
        weighting_method="equal",
        rebalance_frequency=10,  # Rebalance every 10 bars
        rebalance_threshold=0.01,
    )

    backtest_config = BacktestConfig(initial_capital=10000.0)

    portfolio = Portfolio(
        symbols=symbols,
        portfolio_config=portfolio_config,
        backtest_config=backtest_config,
    )

    result = portfolio.run_backtest(
        data_dict=data_dict,
        strategy_func=simple_strategy,
        strategy_params={},
    )

    # Should have rebalanced at least once
    assert len(result.rebalance_dates) > 1
    assert "n_rebalances" in result.metrics


def test_portfolio_get_symbol_statistics():
    """Test getting symbol statistics from portfolio."""
    symbols = ["BTC", "ETH"]
    portfolio = create_portfolio(symbols, weighting_method="equal")

    # Create mock results
    df1 = create_test_dataframe(n_bars=100, base_price=100.0)
    df2 = create_test_dataframe(n_bars=100, base_price=200.0)

    config = BacktestConfig(initial_capital=10000.0)

    result1 = run_single_symbol_backtest("BTC", df1, simple_strategy, {}, config)
    result2 = run_single_symbol_backtest("ETH", df2, simple_strategy, {}, config)

    symbol_results = {"BTC": result1, "ETH": result2}

    stats = portfolio.get_symbol_statistics(symbol_results)

    assert len(stats) == 2
    assert all(col in stats.columns for col in ["symbol", "total_return", "sharpe_ratio"])


def test_portfolio_with_weight_caps():
    """Test portfolio with weight constraints."""
    symbols = ["BTC", "ETH"]
    
    data_dict = {
        "BTC": create_test_dataframe(n_bars=100, base_price=100.0),
        "ETH": create_test_dataframe(n_bars=100, base_price=200.0),
    }

    portfolio_config = PortfolioConfig(
        weighting_method="volatility",
        min_weight=0.3,
        max_weight=0.7,
    )

    backtest_config = BacktestConfig(initial_capital=10000.0)

    portfolio = Portfolio(
        symbols=symbols,
        portfolio_config=portfolio_config,
        backtest_config=backtest_config,
    )

    result = portfolio.run_backtest(
        data_dict=data_dict,
        strategy_func=simple_strategy,
        strategy_params={},
    )

    # Check weights are within bounds
    assert np.all(result.weights >= 0.3)
    assert np.all(result.weights <= 0.7)
    assert np.isclose(result.weights.sum(), 1.0)


def test_portfolio_empty_data():
    """Test portfolio with empty data dict."""
    symbols = ["BTC", "ETH"]
    data_dict = {}

    portfolio_config = PortfolioConfig(weighting_method="equal")
    backtest_config = BacktestConfig(initial_capital=10000.0)

    portfolio = Portfolio(
        symbols=symbols,
        portfolio_config=portfolio_config,
        backtest_config=backtest_config,
    )

    result = portfolio.run_backtest(
        data_dict=data_dict,
        strategy_func=simple_strategy,
        strategy_params={},
    )

    # Should return empty result gracefully
    assert len(result.equity_curve) == 1
    assert result.equity_curve[0] == 10000.0
