"""
Example: Multi-timeframe strategy with enhanced features.

This example demonstrates:
1. Multi-timeframe trend filtering (30m, 1h, 4h)
2. Dynamic position sizing
3. Time-based exits
4. Partial take-profit scaling
5. Optional short trading
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from core.analysis.fourier import smooth_price_series
from core.analysis.mtf import align_timeframes, apply_mtf_filter, check_mtf_alignment, compute_trend_direction
from core.analysis.signals import generate_signals_with_stops
from core.analysis.stops import compute_atr, compute_atr_stops
from core.backtest.engine import BacktestConfig, run_backtest_enhanced, trades_to_dataframe
from core.data.loader import load_klines


def run_mtf_strategy(
    symbol: str = "BTCUSDT",
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    enable_shorts: bool = False,
    use_dynamic_sizing: bool = True,
    use_time_exit: bool = True,
    use_partial_tp: bool = False,
) -> None:
    """
    Run a multi-timeframe strategy with enhanced features.

    Args:
        symbol: Trading symbol
        start_date: Start date for backtest
        end_date: End date for backtest
        enable_shorts: Enable short trading
        use_dynamic_sizing: Use volatility-based position sizing
        use_time_exit: Enable time-based exits
        use_partial_tp: Enable partial take-profit scaling
    """
    if start_date is None:
        start_date = datetime(2024, 1, 1, tzinfo=UTC)
    if end_date is None:
        end_date = datetime(2024, 3, 1, tzinfo=UTC)

    print(f"Loading data for {symbol}...")

    df_30m = load_klines(symbol, "30m", start_date, end_date)
    df_1h = load_klines(symbol, "1h", start_date, end_date)
    df_4h = load_klines(symbol, "4h", start_date, end_date)

    if df_30m.empty:
        print("No data available. Using synthetic data for demonstration.")
        n = 500
        timestamps = pd.date_range(start_date, periods=n, freq="30min")
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df_30m = pd.DataFrame({
            'open_time': timestamps,
            'open': close * (1 + np.random.randn(n) * 0.001),
            'high': close * 1.01,
            'low': close * 0.99,
            'close': close,
        })

        df_1h = df_30m.iloc[::2].reset_index(drop=True)
        df_4h = df_30m.iloc[::8].reset_index(drop=True)

    print(f"Loaded {len(df_30m)} bars on 30m timeframe")

    print("Computing smoothed trends...")
    smoothed_30m = smooth_price_series(df_30m['close'].values, min_period_bars=20)
    smoothed_1h = smooth_price_series(df_1h['close'].values, min_period_bars=10)
    smoothed_4h = smooth_price_series(df_4h['close'].values, min_period_bars=5)

    df_30m['smoothed'] = smoothed_30m
    df_1h['smoothed'] = smoothed_1h
    df_4h['smoothed'] = smoothed_4h

    print("Aligning timeframes...")
    df_aligned = align_timeframes(df_30m, df_1h, "30m", "1h")
    df_aligned = align_timeframes(df_aligned, df_4h, "30m", "4h")

    print("Computing trend directions...")
    trend_30m = compute_trend_direction(
        df_aligned['close'].values,
        df_aligned['smoothed'].values,
        slope_lookback=2
    )

    trend_1h = compute_trend_direction(
        df_aligned['close'].values,
        df_aligned['smoothed_1h'].values,
        slope_lookback=2
    )

    trend_4h = compute_trend_direction(
        df_aligned['close'].values,
        df_aligned['smoothed_4h'].values,
        slope_lookback=2
    )

    aligned_long, aligned_short = check_mtf_alignment(
        trend_30m, trend_1h, trend_4h, require_all=False
    )

    print(f"MTF aligned long opportunities: {aligned_long.sum()}")
    print(f"MTF aligned short opportunities: {aligned_short.sum()}")

    print("Computing stops and signals...")
    long_stop, long_profit, short_stop, short_profit = compute_atr_stops(
        df_aligned['close'].values,
        df_aligned['high'].values,
        df_aligned['low'].values,
        atr_period=14,
        k_stop=2.0,
        k_profit=3.0,
    )

    signals = generate_signals_with_stops(
        close=df_aligned['close'].values,
        smoothed=df_aligned['smoothed'].values,
        stop_levels=long_stop,
        slope_threshold=0.0,
        slope_lookback=2,
    )

    signals_filtered = apply_mtf_filter(signals, aligned_long, direction=1)

    if enable_shorts:
        short_signals = generate_signals_with_stops(
            close=df_aligned['close'].values,
            smoothed=df_aligned['smoothed'].values,
            stop_levels=short_stop,
            slope_threshold=0.0,
            slope_lookback=2,
        )
        short_signals_filtered = apply_mtf_filter(short_signals, aligned_short, direction=-1)

        signals_combined = signals_filtered.copy()
        signals_combined[short_signals_filtered == 1] = 2
        signals_combined[short_signals_filtered == -1] = -2
        signals_filtered = signals_combined

    print("Running backtest...")

    atr = compute_atr(
        df_aligned['high'].values,
        df_aligned['low'].values,
        df_aligned['close'].values,
        period=14,
    )

    config = BacktestConfig(
        initial_capital=10000.0,
        fee_rate=0.001,
        slippage=0.0005,
        allow_shorts=enable_shorts,
        max_bars_held=100 if use_time_exit else None,
        enable_partial_tp=use_partial_tp,
        partial_tp_scales=[(0.02, 0.5), (0.05, 0.3)] if use_partial_tp else None,
        sizing_mode="volatility" if use_dynamic_sizing else "fixed",
        volatility_target=0.02,
        max_risk_per_trade=0.02,
    )

    result = run_backtest_enhanced(
        signals=signals_filtered,
        open_prices=df_aligned['open'].values,
        high_prices=df_aligned['high'].values,
        low_prices=df_aligned['low'].values,
        close_prices=df_aligned['close'].values,
        timestamps=df_aligned['open_time'],
        atr=atr,
        stop_levels=long_stop,
        config=config,
    )

    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)

    print(f"\nTotal Return: {result.metrics['total_return']:.2%}")
    print(f"Annualized Return: {result.metrics['annualized_return']:.2%}")
    print(f"Max Drawdown: {result.metrics['max_drawdown_pct']:.2%}")
    print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {result.metrics['sortino_ratio']:.2f}")

    print(f"\nNumber of Trades: {result.metrics['n_trades']}")
    print(f"Win Rate: {result.metrics['win_rate']:.2%}")
    print(f"Profit Factor: {result.metrics['profit_factor']:.2f}")
    print(f"Average Win: ${result.metrics['avg_win']:.2f}")
    print(f"Average Loss: ${result.metrics['avg_loss']:.2f}")
    print(f"Average Bars Held: {result.metrics['avg_bars_held']:.1f}")

    if result.trades:
        print("\n" + "-"*60)
        print("SAMPLE TRADES")
        print("-"*60)
        trades_df = trades_to_dataframe(result.trades)
        print(trades_df.head(10).to_string(index=False))

        if enable_shorts:
            long_trades = sum(1 for t in result.trades if t.direction == 1)
            short_trades = sum(1 for t in result.trades if t.direction == -1)
            print(f"\nLong trades: {long_trades}")
            print(f"Short trades: {short_trades}")

        exit_reasons = {}
        for trade in result.trades:
            exit_reasons[trade.exit_reason] = exit_reasons.get(trade.exit_reason, 0) + 1
        print("\nExit reasons:")
        for reason, count in exit_reasons.items():
            print(f"  {reason}: {count}")

    print("\n" + "="*60)


if __name__ == "__main__":
    print("Multi-Timeframe Strategy Example\n")

    print("Configuration:")
    print("- Timeframes: 30m (execution), 1h, 4h (trend filters)")
    print("- Position sizing: Volatility-based (ATR)")
    print("- Exits: Stop loss, time-based, signal-based")
    print("- Shorts: Optional (disabled by default)")
    print()

    run_mtf_strategy(
        symbol="BTCUSDT",
        enable_shorts=False,
        use_dynamic_sizing=True,
        use_time_exit=True,
        use_partial_tp=False,
    )
