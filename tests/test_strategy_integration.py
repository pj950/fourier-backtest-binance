import numpy as np
import pandas as pd
import pytest

from core.analysis.fourier import smooth_price_series
from core.analysis.mtf import align_timeframes, check_mtf_alignment, compute_trend_direction
from core.analysis.signals import generate_signals_with_stops
from core.analysis.stops import compute_atr_stops
from core.backtest.engine import BacktestConfig, run_backtest, run_backtest_enhanced


def test_mtf_strategy_integration():
    """Test multi-timeframe strategy with trend filtering."""
    n = 200

    close_30m = np.linspace(100, 120, n)
    close_30m += np.sin(np.linspace(0, 10, n)) * 2

    smoothed_30m = smooth_price_series(close_30m, min_period_bars=10)
    trend_30m = compute_trend_direction(close_30m, smoothed_30m)

    trend_1h = np.ones(n, dtype=int)
    trend_1h[:50] = 0

    trend_4h = np.ones(n, dtype=int)

    aligned_long, _ = check_mtf_alignment(trend_30m, trend_1h, trend_4h, require_all=True)

    assert np.any(aligned_long)
    assert not aligned_long[0:50].any()


def test_enhanced_backtest_with_mtf():
    """Test enhanced backtest with MTF filtering."""
    n = 100
    signals = np.zeros(n, dtype=int)
    signals[10] = 1
    signals[20] = -1
    signals[30] = 1
    signals[40] = -1

    open_prices = np.linspace(100, 110, n)
    high_prices = open_prices * 1.01
    low_prices = open_prices * 0.99
    close_prices = open_prices
    timestamps = pd.date_range('2024-01-01', periods=n, freq='30min')

    config = BacktestConfig(initial_capital=10000.0)

    result = run_backtest_enhanced(
        signals, open_prices, high_prices, low_prices, close_prices, timestamps, config=config
    )

    assert len(result.trades) >= 2
    assert result.metrics['n_trades'] >= 2


def test_volatility_sizing_with_stops():
    """Test dynamic position sizing with ATR-based stops."""
    n = 100
    signals = np.zeros(n, dtype=int)
    signals[10] = 1
    signals[30] = -1

    open_prices = np.full(n, 100.0)
    high_prices = np.full(n, 102.0)
    low_prices = np.full(n, 98.0)
    close_prices = open_prices
    timestamps = pd.date_range('2024-01-01', periods=n, freq='1h')

    long_stop, _, _, _ = compute_atr_stops(
        close_prices, high_prices, low_prices, atr_period=14, k_stop=2.0
    )

    from core.analysis.stops import compute_atr
    atr = compute_atr(high_prices, low_prices, close_prices, period=14)

    config = BacktestConfig(
        initial_capital=10000.0,
        sizing_mode="volatility",
        volatility_target=0.02,
        max_risk_per_trade=0.03,
    )

    result = run_backtest_enhanced(
        signals, open_prices, high_prices, low_prices, close_prices, timestamps,
        atr=atr, stop_levels=long_stop, config=config
    )

    assert len(result.trades) == 1
    assert result.trades[0].size > 0


def test_time_based_exit_integration():
    """Test time-based exit integration."""
    n = 100
    signals = np.zeros(n, dtype=int)
    signals[10] = 1

    open_prices = np.full(n, 100.0)
    high_prices = open_prices * 1.01
    low_prices = open_prices * 0.99
    close_prices = open_prices
    timestamps = pd.date_range('2024-01-01', periods=n, freq='1h')

    config = BacktestConfig(initial_capital=10000.0, max_bars_held=20)

    result = run_backtest_enhanced(
        signals, open_prices, high_prices, low_prices, close_prices, timestamps, config=config
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "time"
    assert result.trades[0].bars_held == 20


def test_short_trading_integration():
    """Test short trading with futures flag."""
    n = 100
    signals = np.zeros(n, dtype=int)
    signals[10] = 2
    signals[30] = -2

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


def test_full_strategy_workflow():
    """Test complete strategy workflow with all features."""
    n = 300

    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close * 1.01
    low = close * 0.99
    open_prices = close * (1 + np.random.randn(n) * 0.001)

    smoothed = smooth_price_series(close, min_period_bars=20)

    long_stop, long_profit, _, _ = compute_atr_stops(
        close, high, low, atr_period=14, k_stop=2.0, k_profit=3.0
    )

    signals = generate_signals_with_stops(
        close, smoothed, long_stop, slope_threshold=0.0, slope_lookback=1
    )

    timestamps = pd.date_range('2024-01-01', periods=n, freq='1h')

    config = BacktestConfig(
        initial_capital=10000.0,
        fee_rate=0.001,
        slippage=0.0005,
        max_bars_held=50,
    )

    result = run_backtest_enhanced(
        signals, open_prices, high, low, close, timestamps, config=config
    )

    assert len(result.equity_curve) == n
    assert 'total_return' in result.metrics
    assert 'sharpe_ratio' in result.metrics
    assert result.metrics['n_trades'] >= 0
