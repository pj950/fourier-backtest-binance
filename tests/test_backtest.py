import numpy as np
import pandas as pd
import pytest

from core.backtest.engine import (
    BacktestConfig,
    compute_mae_mfe,
    compute_max_drawdown,
    compute_metrics,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    run_backtest,
    trades_to_dataframe,
)


@pytest.fixture
def fixture_data() -> dict:
    """Fixture data for deterministic backtesting."""
    n = 100
    np.random.seed(42)

    timestamps = pd.date_range(start="2024-01-01", periods=n, freq="1h")
    base_price = 100.0
    trend = np.linspace(0, 10, n)
    noise = np.random.randn(n) * 0.5

    close = base_price + trend + noise
    open_prices = close + np.random.randn(n) * 0.2
    high = np.maximum(close, open_prices) + np.abs(np.random.randn(n)) * 0.3
    low = np.minimum(close, open_prices) - np.abs(np.random.randn(n)) * 0.3

    return {
        "timestamps": timestamps,
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
    }


def test_compute_mae_mfe_long() -> None:
    """Test MAE/MFE computation for long position."""
    entry_price = 100.0
    high = np.array([100, 102, 105, 103, 101])
    low = np.array([99, 100, 102, 101, 99])
    entry_idx = 0
    exit_idx = 4

    mae, mfe = compute_mae_mfe(entry_price, high, low, entry_idx, exit_idx, direction=1)

    assert mae == 1.0
    assert mfe == 5.0


def test_compute_mae_mfe_short() -> None:
    """Test MAE/MFE computation for short position."""
    entry_price = 100.0
    high = np.array([100, 102, 105, 103, 101])
    low = np.array([99, 100, 102, 101, 99])
    entry_idx = 0
    exit_idx = 4

    mae, mfe = compute_mae_mfe(entry_price, high, low, entry_idx, exit_idx, direction=-1)

    assert mae == 5.0
    assert mfe == 1.0


def test_compute_max_drawdown() -> None:
    """Test maximum drawdown computation."""
    equity = np.array([100, 110, 105, 115, 100, 120])

    max_dd, max_dd_pct = compute_max_drawdown(equity)

    assert max_dd < 0
    assert max_dd_pct < 0


def test_compute_max_drawdown_no_drawdown() -> None:
    """Test max drawdown with monotonically increasing equity."""
    equity = np.array([100, 110, 120, 130, 140])

    max_dd, max_dd_pct = compute_max_drawdown(equity)

    assert max_dd == 0.0
    assert max_dd_pct == 0.0


def test_compute_sharpe_ratio() -> None:
    """Test Sharpe ratio computation."""
    np.random.seed(42)
    returns = np.random.randn(100) * 0.01 + 0.001

    sharpe = compute_sharpe_ratio(returns, periods_per_year=252)

    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)


def test_compute_sharpe_ratio_zero_std() -> None:
    """Test Sharpe ratio with zero standard deviation."""
    returns = np.ones(100) * 0.01

    sharpe = compute_sharpe_ratio(returns)

    assert sharpe == 0.0


def test_compute_sortino_ratio() -> None:
    """Test Sortino ratio computation."""
    np.random.seed(42)
    returns = np.random.randn(100) * 0.01 + 0.001

    sortino = compute_sortino_ratio(returns, periods_per_year=252)

    assert isinstance(sortino, float)
    assert not np.isnan(sortino)


def test_run_backtest_basic(fixture_data: dict) -> None:
    """Test basic backtest execution."""
    signals = np.zeros(len(fixture_data["close"]), dtype=int)
    signals[10] = 1
    signals[20] = -1
    signals[30] = 1
    signals[40] = -1

    config = BacktestConfig(initial_capital=10000.0, fee_rate=0.001, slippage=0.0005)

    result = run_backtest(
        signals=signals,
        open_prices=fixture_data["open"],
        high_prices=fixture_data["high"],
        low_prices=fixture_data["low"],
        close_prices=fixture_data["close"],
        timestamps=fixture_data["timestamps"],
        config=config,
    )

    assert len(result.equity_curve) == len(signals)
    assert result.equity_curve[0] == config.initial_capital
    assert len(result.trades) == 2
    assert len(result.metrics) > 0


def test_run_backtest_no_signals(fixture_data: dict) -> None:
    """Test backtest with no signals."""
    signals = np.zeros(len(fixture_data["close"]), dtype=int)

    result = run_backtest(
        signals=signals,
        open_prices=fixture_data["open"],
        high_prices=fixture_data["high"],
        low_prices=fixture_data["low"],
        close_prices=fixture_data["close"],
        timestamps=fixture_data["timestamps"],
    )

    assert len(result.trades) == 0
    assert result.equity_curve[-1] == result.equity_curve[0]


def test_run_backtest_deterministic(fixture_data: dict) -> None:
    """Test that backtest is deterministic with same inputs."""
    signals = np.zeros(len(fixture_data["close"]), dtype=int)
    signals[10] = 1
    signals[20] = -1

    result1 = run_backtest(
        signals=signals,
        open_prices=fixture_data["open"],
        high_prices=fixture_data["high"],
        low_prices=fixture_data["low"],
        close_prices=fixture_data["close"],
        timestamps=fixture_data["timestamps"],
    )

    result2 = run_backtest(
        signals=signals,
        open_prices=fixture_data["open"],
        high_prices=fixture_data["high"],
        low_prices=fixture_data["low"],
        close_prices=fixture_data["close"],
        timestamps=fixture_data["timestamps"],
    )

    assert np.array_equal(result1.equity_curve, result2.equity_curve)
    assert len(result1.trades) == len(result2.trades)


def test_run_backtest_with_fees(fixture_data: dict) -> None:
    """Test that fees reduce profits."""
    signals = np.zeros(len(fixture_data["close"]), dtype=int)
    signals[10] = 1
    signals[20] = -1

    config_no_fees = BacktestConfig(fee_rate=0.0, slippage=0.0)
    config_with_fees = BacktestConfig(fee_rate=0.001, slippage=0.0)

    result_no_fees = run_backtest(
        signals=signals,
        open_prices=fixture_data["open"],
        high_prices=fixture_data["high"],
        low_prices=fixture_data["low"],
        close_prices=fixture_data["close"],
        timestamps=fixture_data["timestamps"],
        config=config_no_fees,
    )

    result_with_fees = run_backtest(
        signals=signals,
        open_prices=fixture_data["open"],
        high_prices=fixture_data["high"],
        low_prices=fixture_data["low"],
        close_prices=fixture_data["close"],
        timestamps=fixture_data["timestamps"],
        config=config_with_fees,
    )

    assert result_no_fees.equity_curve[-1] >= result_with_fees.equity_curve[-1]


def test_run_backtest_position_sizing_fraction(fixture_data: dict) -> None:
    """Test fractional position sizing."""
    signals = np.zeros(len(fixture_data["close"]), dtype=int)
    signals[10] = 1
    signals[20] = -1

    config_full = BacktestConfig(position_size_mode="full", position_size_fraction=1.0)
    config_half = BacktestConfig(position_size_mode="full", position_size_fraction=0.5)

    result_full = run_backtest(
        signals=signals,
        open_prices=fixture_data["open"],
        high_prices=fixture_data["high"],
        low_prices=fixture_data["low"],
        close_prices=fixture_data["close"],
        timestamps=fixture_data["timestamps"],
        config=config_full,
    )

    result_half = run_backtest(
        signals=signals,
        open_prices=fixture_data["open"],
        high_prices=fixture_data["high"],
        low_prices=fixture_data["low"],
        close_prices=fixture_data["close"],
        timestamps=fixture_data["timestamps"],
        config=config_half,
    )

    assert result_full.trades[0].size > result_half.trades[0].size


def test_compute_metrics_complete(fixture_data: dict) -> None:
    """Test that all expected metrics are computed."""
    signals = np.zeros(len(fixture_data["close"]), dtype=int)
    signals[10] = 1
    signals[20] = -1
    signals[30] = 1
    signals[40] = -1

    result = run_backtest(
        signals=signals,
        open_prices=fixture_data["open"],
        high_prices=fixture_data["high"],
        low_prices=fixture_data["low"],
        close_prices=fixture_data["close"],
        timestamps=fixture_data["timestamps"],
    )

    expected_metrics = [
        "total_return",
        "cumulative_return",
        "annualized_return",
        "max_drawdown",
        "max_drawdown_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "n_trades",
        "n_wins",
        "n_losses",
        "win_rate",
        "profit_factor",
        "avg_win",
        "avg_loss",
        "avg_bars_held",
        "avg_mae",
        "avg_mfe",
        "avg_mae_pct",
        "avg_mfe_pct",
    ]

    for metric in expected_metrics:
        assert metric in result.metrics


def test_trades_to_dataframe(fixture_data: dict) -> None:
    """Test conversion of trades to DataFrame."""
    signals = np.zeros(len(fixture_data["close"]), dtype=int)
    signals[10] = 1
    signals[20] = -1

    result = run_backtest(
        signals=signals,
        open_prices=fixture_data["open"],
        high_prices=fixture_data["high"],
        low_prices=fixture_data["low"],
        close_prices=fixture_data["close"],
        timestamps=fixture_data["timestamps"],
    )

    df = trades_to_dataframe(result.trades)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(result.trades)
    assert "entry_price" in df.columns
    assert "exit_price" in df.columns
    assert "pnl" in df.columns


def test_trades_to_dataframe_empty() -> None:
    """Test conversion of empty trades list."""
    df = trades_to_dataframe([])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
