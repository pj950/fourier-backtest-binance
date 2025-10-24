import numpy as np
import pandas as pd
import pytest

from core.backtest.engine import BacktestConfig, run_backtest_enhanced


def test_backtest_enhanced_basic_long():
    n = 100
    signals = np.zeros(n, dtype=int)
    signals[10] = 1
    signals[20] = -1

    open_prices = np.linspace(100, 110, n)
    high_prices = open_prices * 1.01
    low_prices = open_prices * 0.99
    close_prices = open_prices
    timestamps = pd.date_range('2024-01-01', periods=n, freq='1h')

    config = BacktestConfig(initial_capital=10000.0)

    result = run_backtest_enhanced(
        signals, open_prices, high_prices, low_prices, close_prices, timestamps, config=config
    )

    assert len(result.trades) == 1
    assert result.trades[0].direction == 1
    assert result.trades[0].entry_idx == 11
    assert result.trades[0].exit_idx == 21


def test_backtest_enhanced_with_shorts():
    n = 100
    signals = np.zeros(n, dtype=int)
    signals[10] = 2
    signals[20] = -2

    open_prices = np.linspace(100, 90, n)
    high_prices = open_prices * 1.01
    low_prices = open_prices * 0.99
    close_prices = open_prices
    timestamps = pd.date_range('2024-01-01', periods=n, freq='1h')

    config = BacktestConfig(initial_capital=10000.0, allow_shorts=True)

    result = run_backtest_enhanced(
        signals, open_prices, high_prices, low_prices, close_prices, timestamps, config=config
    )

    assert len(result.trades) == 1
    assert result.trades[0].direction == -1
    assert result.trades[0].pnl > 0


def test_backtest_enhanced_time_based_exit():
    n = 100
    signals = np.zeros(n, dtype=int)
    signals[10] = 1

    open_prices = np.full(n, 100.0)
    high_prices = open_prices * 1.01
    low_prices = open_prices * 0.99
    close_prices = open_prices
    timestamps = pd.date_range('2024-01-01', periods=n, freq='1h')

    config = BacktestConfig(initial_capital=10000.0, max_bars_held=15)

    result = run_backtest_enhanced(
        signals, open_prices, high_prices, low_prices, close_prices, timestamps, config=config
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "time"
    assert result.trades[0].bars_held == 15


def test_backtest_enhanced_stop_loss():
    n = 100
    signals = np.zeros(n, dtype=int)
    signals[10] = 1

    open_prices = np.linspace(100, 90, n)
    high_prices = open_prices * 1.01
    low_prices = open_prices * 0.99
    close_prices = open_prices
    timestamps = pd.date_range('2024-01-01', periods=n, freq='1h')

    stop_levels = np.full(n, 95.0)

    config = BacktestConfig(initial_capital=10000.0)

    result = run_backtest_enhanced(
        signals, open_prices, high_prices, low_prices, close_prices, timestamps,
        stop_levels=stop_levels, config=config
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "stop"
    assert result.trades[0].pnl < 0


def test_backtest_enhanced_volatility_sizing():
    n = 100
    signals = np.zeros(n, dtype=int)
    signals[10] = 1
    signals[20] = -1

    open_prices = np.full(n, 100.0)
    high_prices = open_prices * 1.01
    low_prices = open_prices * 0.99
    close_prices = open_prices
    timestamps = pd.date_range('2024-01-01', periods=n, freq='1h')

    atr = np.full(n, 2.0)
    stop_levels = np.full(n, 95.0)

    config = BacktestConfig(
        initial_capital=10000.0,
        sizing_mode="volatility",
        volatility_target=0.02,
        max_risk_per_trade=0.05,
    )

    result = run_backtest_enhanced(
        signals, open_prices, high_prices, low_prices, close_prices, timestamps,
        atr=atr, stop_levels=stop_levels, config=config
    )

    assert len(result.trades) == 1
    assert result.trades[0].size > 0


def test_backtest_enhanced_fixed_risk_sizing():
    n = 100
    signals = np.zeros(n, dtype=int)
    signals[10] = 1
    signals[20] = -1

    open_prices = np.full(n, 100.0)
    high_prices = open_prices * 1.01
    low_prices = open_prices * 0.99
    close_prices = open_prices
    timestamps = pd.date_range('2024-01-01', periods=n, freq='1h')

    stop_levels = np.full(n, 95.0)

    config = BacktestConfig(
        initial_capital=10000.0,
        sizing_mode="fixed_risk",
        max_risk_per_trade=0.01,
    )

    result = run_backtest_enhanced(
        signals, open_prices, high_prices, low_prices, close_prices, timestamps,
        stop_levels=stop_levels, config=config
    )

    assert len(result.trades) == 1

    expected_size = (10000.0 * 0.01) / 5.0
    actual_size = result.trades[0].size
    assert abs(actual_size - expected_size) < expected_size * 0.1


def test_backtest_enhanced_multiple_trades():
    n = 100
    signals = np.zeros(n, dtype=int)
    signals[10] = 1
    signals[20] = -1
    signals[30] = 1
    signals[40] = -1
    signals[50] = 1
    signals[60] = -1

    open_prices = np.linspace(100, 120, n)
    high_prices = open_prices * 1.01
    low_prices = open_prices * 0.99
    close_prices = open_prices
    timestamps = pd.date_range('2024-01-01', periods=n, freq='1h')

    config = BacktestConfig(initial_capital=10000.0)

    result = run_backtest_enhanced(
        signals, open_prices, high_prices, low_prices, close_prices, timestamps, config=config
    )

    assert len(result.trades) == 3
    assert all(t.direction == 1 for t in result.trades)


def test_backtest_enhanced_no_trades():
    n = 100
    signals = np.zeros(n, dtype=int)

    open_prices = np.linspace(100, 110, n)
    high_prices = open_prices * 1.01
    low_prices = open_prices * 0.99
    close_prices = open_prices
    timestamps = pd.date_range('2024-01-01', periods=n, freq='1h')

    config = BacktestConfig(initial_capital=10000.0)

    result = run_backtest_enhanced(
        signals, open_prices, high_prices, low_prices, close_prices, timestamps, config=config
    )

    assert len(result.trades) == 0
    assert result.equity_curve[-1] == config.initial_capital


def test_backtest_enhanced_partial_tp():
    n = 100
    signals = np.zeros(n, dtype=int)
    signals[10] = 1

    open_prices = np.full(10, 100.0)
    open_prices = np.concatenate([open_prices, np.linspace(100, 110, n - 10)])
    high_prices = open_prices * 1.02
    low_prices = open_prices * 0.99
    close_prices = open_prices
    timestamps = pd.date_range('2024-01-01', periods=n, freq='1h')

    partial_tp_scales = [(0.02, 0.5), (0.05, 0.3)]

    config = BacktestConfig(
        initial_capital=10000.0,
        enable_partial_tp=True,
        partial_tp_scales=partial_tp_scales,
    )

    result = run_backtest_enhanced(
        signals, open_prices, high_prices, low_prices, close_prices, timestamps, config=config
    )

    assert len(result.trades) <= 1


def test_backtest_enhanced_metrics():
    n = 100
    signals = np.zeros(n, dtype=int)
    signals[10] = 1
    signals[20] = -1

    open_prices = np.linspace(100, 110, n)
    high_prices = open_prices * 1.01
    low_prices = open_prices * 0.99
    close_prices = open_prices
    timestamps = pd.date_range('2024-01-01', periods=n, freq='1h')

    config = BacktestConfig(initial_capital=10000.0)

    result = run_backtest_enhanced(
        signals, open_prices, high_prices, low_prices, close_prices, timestamps, config=config
    )

    assert 'total_return' in result.metrics
    assert 'sharpe_ratio' in result.metrics
    assert 'n_trades' in result.metrics
    assert result.metrics['n_trades'] == 1
