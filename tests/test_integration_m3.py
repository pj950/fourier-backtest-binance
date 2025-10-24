"""Integration test for M3 components: signals, stops, and backtester."""

import numpy as np
import pandas as pd
import pytest

from core.analysis.fourier import smooth_price_series
from core.analysis.signals import generate_signals_with_stops
from core.analysis.stops import compute_atr, compute_atr_stops, compute_residual_stops
from core.backtest.engine import BacktestConfig, run_backtest, trades_to_dataframe


@pytest.fixture
def synthetic_market_data() -> dict:
    """Generate synthetic market data with trends and noise."""
    np.random.seed(123)
    n = 500

    timestamps = pd.date_range(start="2024-01-01", periods=n, freq="1h")

    base = 10000.0
    trend = np.cumsum(np.random.randn(n) * 5)
    noise = np.random.randn(n) * 20

    close = base + trend + noise

    open_prices = close + np.random.randn(n) * 5
    high = np.maximum(close, open_prices) + np.abs(np.random.randn(n)) * 10
    low = np.minimum(close, open_prices) - np.abs(np.random.randn(n)) * 10

    return {
        "timestamps": timestamps,
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
    }


def test_complete_workflow_atr_stops(synthetic_market_data: dict) -> None:
    """Test complete workflow with ATR-based stops."""
    close = synthetic_market_data["close"]
    high = synthetic_market_data["high"]
    low = synthetic_market_data["low"]
    open_prices = synthetic_market_data["open"]
    timestamps = synthetic_market_data["timestamps"]

    smoothed = smooth_price_series(close, min_period_bars=24, cutoff_scale=1.0)

    atr = compute_atr(high, low, close, period=14)
    long_stop, long_profit, short_stop, short_profit = compute_atr_stops(
        close, high, low, atr_period=14, k_stop=2.0, k_profit=3.0
    )

    signals = generate_signals_with_stops(
        close=close,
        smoothed=smoothed,
        stop_levels=long_stop,
        slope_threshold=0.0,
        slope_lookback=1,
    )

    config = BacktestConfig(initial_capital=100000.0, fee_rate=0.001, slippage=0.0005)

    result = run_backtest(
        signals=signals,
        open_prices=open_prices,
        high_prices=high,
        low_prices=low,
        close_prices=close,
        timestamps=timestamps,
        config=config,
    )

    assert len(result.equity_curve) == len(close)
    assert result.equity_curve[0] == config.initial_capital
    assert len(result.metrics) > 0

    print(f"\n=== ATR-Based Backtest Results ===")
    print(f"Number of trades: {result.metrics['n_trades']}")
    print(f"Win rate: {result.metrics['win_rate']:.2%}")
    print(f"Total return: {result.metrics['total_return']:.2%}")
    print(f"Max drawdown: {result.metrics['max_drawdown_pct']:.2%}")
    print(f"Sharpe ratio: {result.metrics['sharpe_ratio']:.2f}")


def test_complete_workflow_residual_stops(synthetic_market_data: dict) -> None:
    """Test complete workflow with residual-based stops."""
    close = synthetic_market_data["close"]
    high = synthetic_market_data["high"]
    low = synthetic_market_data["low"]
    open_prices = synthetic_market_data["open"]
    timestamps = synthetic_market_data["timestamps"]

    smoothed = smooth_price_series(close, min_period_bars=24, cutoff_scale=1.0)

    long_stop, long_profit, short_stop, short_profit = compute_residual_stops(
        close=close,
        smoothed=smoothed,
        method="sigma",
        window=20,
        k_stop=2.0,
        k_profit=3.0,
    )

    signals = generate_signals_with_stops(
        close=close,
        smoothed=smoothed,
        stop_levels=long_stop,
        slope_threshold=0.0,
        slope_lookback=1,
    )

    config = BacktestConfig(initial_capital=100000.0, fee_rate=0.001, slippage=0.0005)

    result = run_backtest(
        signals=signals,
        open_prices=open_prices,
        high_prices=high,
        low_prices=low,
        close_prices=close,
        timestamps=timestamps,
        config=config,
    )

    assert len(result.equity_curve) == len(close)
    assert len(result.metrics) > 0

    df_trades = trades_to_dataframe(result.trades)

    print(f"\n=== Residual-Based Backtest Results ===")
    print(f"Number of trades: {result.metrics['n_trades']}")
    print(f"Win rate: {result.metrics['win_rate']:.2%}")
    print(f"Total return: {result.metrics['total_return']:.2%}")
    print(f"Max drawdown: {result.metrics['max_drawdown_pct']:.2%}")
    print(f"Sharpe ratio: {result.metrics['sharpe_ratio']:.2f}")
    print(f"Avg MAE: {result.metrics['avg_mae_pct']:.2%}")
    print(f"Avg MFE: {result.metrics['avg_mfe_pct']:.2%}")

    if len(df_trades) > 0:
        print(f"\nFirst 3 trades:")
        print(df_trades[["entry_time", "exit_time", "pnl", "pnl_pct", "bars_held"]].head(3))


def test_metrics_calculation(synthetic_market_data: dict) -> None:
    """Test that all required metrics are calculated."""
    close = synthetic_market_data["close"]
    high = synthetic_market_data["high"]
    low = synthetic_market_data["low"]
    open_prices = synthetic_market_data["open"]
    timestamps = synthetic_market_data["timestamps"]

    smoothed = smooth_price_series(close, min_period_bars=24)
    long_stop, _, _, _ = compute_atr_stops(close, high, low)

    signals = generate_signals_with_stops(close, smoothed, long_stop)

    result = run_backtest(
        signals, open_prices, high, low, close, timestamps
    )

    required_metrics = [
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

    for metric in required_metrics:
        assert metric in result.metrics, f"Missing metric: {metric}"
        assert not np.isnan(result.metrics[metric]), f"Metric {metric} is NaN"


def test_deterministic_backtest(synthetic_market_data: dict) -> None:
    """Test that backtest results are deterministic."""
    close = synthetic_market_data["close"]
    high = synthetic_market_data["high"]
    low = synthetic_market_data["low"]
    open_prices = synthetic_market_data["open"]
    timestamps = synthetic_market_data["timestamps"]

    smoothed = smooth_price_series(close, min_period_bars=24, cutoff_scale=1.0)
    long_stop, _, _, _ = compute_atr_stops(close, high, low, atr_period=14)
    signals = generate_signals_with_stops(close, smoothed, long_stop)

    config = BacktestConfig(initial_capital=100000.0, fee_rate=0.001)

    result1 = run_backtest(signals, open_prices, high, low, close, timestamps, config)
    result2 = run_backtest(signals, open_prices, high, low, close, timestamps, config)

    assert np.array_equal(result1.equity_curve, result2.equity_curve)
    assert len(result1.trades) == len(result2.trades)
    assert result1.metrics["total_return"] == result2.metrics["total_return"]


def test_parameter_adjustability(synthetic_market_data: dict) -> None:
    """Test that changing parameters affects backtest results."""
    close = synthetic_market_data["close"]
    high = synthetic_market_data["high"]
    low = synthetic_market_data["low"]
    open_prices = synthetic_market_data["open"]
    timestamps = synthetic_market_data["timestamps"]

    smoothed = smooth_price_series(close, min_period_bars=24)

    long_stop_tight, _, _, _ = compute_atr_stops(
        close, high, low, atr_period=14, k_stop=1.0
    )
    long_stop_wide, _, _, _ = compute_atr_stops(
        close, high, low, atr_period=14, k_stop=3.0
    )

    signals_tight = generate_signals_with_stops(close, smoothed, long_stop_tight)
    signals_wide = generate_signals_with_stops(close, smoothed, long_stop_wide)

    result_tight = run_backtest(
        signals_tight, open_prices, high, low, close, timestamps
    )
    result_wide = run_backtest(
        signals_wide, open_prices, high, low, close, timestamps
    )

    print(f"\n=== Parameter Sensitivity ===")
    print(f"Tight stops (k=1.0): {result_tight.metrics['n_trades']} trades")
    print(f"Wide stops (k=3.0): {result_wide.metrics['n_trades']} trades")
