"""
Portfolio Management Example

Demonstrates multi-symbol portfolio backtesting with different weighting schemes.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from core.analysis.fourier import smooth_price_series
from core.analysis.signals import generate_signals_with_stops
from core.analysis.stops import compute_atr_stops
from core.backtest.engine import BacktestConfig
from core.data.loader import load_klines
from core.portfolio.portfolio import Portfolio, PortfolioConfig


def simple_trend_strategy(df: pd.DataFrame, **params) -> np.ndarray:
    """
    Simple trend-following strategy with ATR stops.

    Args:
        df: OHLCV DataFrame
        **params: Strategy parameters

    Returns:
        Signal array (1=entry, -1=exit, 0=hold)
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # Smooth prices
    min_period_bars = params.get("min_period_bars", 24)
    cutoff_scale = params.get("cutoff_scale", 1.0)

    smoothed = smooth_price_series(
        close,
        min_period_bars=min_period_bars,
        cutoff_scale=cutoff_scale,
    )

    # Compute stops
    atr_period = params.get("atr_period", 14)
    k_stop = params.get("k_stop", 2.0)

    long_stop, _, _, _ = compute_atr_stops(
        close=close,
        high=high,
        low=low,
        atr_period=atr_period,
        k_stop=k_stop,
        k_profit=3.0,
    )

    # Generate signals
    signals = generate_signals_with_stops(
        close=close,
        smoothed=smoothed,
        stop_levels=long_stop,
        slope_threshold=0.0,
        slope_lookback=1,
    )

    return signals


def main():
    """Run portfolio backtest example."""
    print("=" * 80)
    print("Portfolio Management Example")
    print("=" * 80)

    # Configuration
    symbols = ["BTCUSDT", "ETHUSDT"]
    interval = "1h"
    start_date = datetime(2024, 1, 1, tzinfo=UTC)
    end_date = datetime(2024, 6, 1, tzinfo=UTC)

    print(f"\nSymbols: {symbols}")
    print(f"Interval: {interval}")
    print(f"Period: {start_date.date()} to {end_date.date()}")

    # Load data for all symbols
    print("\n" + "=" * 80)
    print("Loading Data")
    print("=" * 80)

    data_dict = {}
    for symbol in symbols:
        print(f"\nLoading {symbol}...")
        df = load_klines(
            symbol=symbol,
            interval=interval,
            start=start_date,
            end=end_date,
        )
        print(f"  Loaded {len(df)} bars")
        data_dict[symbol] = df

    # Strategy parameters
    strategy_params = {
        "min_period_bars": 24,
        "cutoff_scale": 1.0,
        "atr_period": 14,
        "k_stop": 2.0,
    }

    print("\n" + "=" * 80)
    print("Strategy Parameters")
    print("=" * 80)
    for key, value in strategy_params.items():
        print(f"  {key}: {value}")

    # Test different weighting schemes
    weighting_methods = ["equal", "volatility", "risk_parity"]

    for method in weighting_methods:
        print("\n" + "=" * 80)
        print(f"Running Portfolio Backtest: {method.upper()} Weights")
        print("=" * 80)

        # Configure portfolio
        portfolio_config = PortfolioConfig(
            weighting_method=method,
            rebalance_frequency=24,  # Daily rebalancing
            rebalance_threshold=0.05,
            min_weight=0.0,
            max_weight=1.0,
            target_vol=0.02,
            lookback_period=60,
        )

        backtest_config = BacktestConfig(
            initial_capital=10000.0,
            fee_rate=0.001,
            slippage=0.0005,
        )

        # Create portfolio
        portfolio = Portfolio(
            symbols=symbols,
            portfolio_config=portfolio_config,
            backtest_config=backtest_config,
        )

        # Run backtest
        result = portfolio.run_backtest(
            data_dict=data_dict,
            strategy_func=simple_trend_strategy,
            strategy_params=strategy_params,
        )

        # Display results
        print("\nPortfolio Metrics:")
        print("-" * 40)
        print(f"  Total Return:        {result.metrics['total_return'] * 100:.2f}%")
        print(f"  Sharpe Ratio:        {result.metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:        {result.metrics['max_drawdown_pct'] * 100:.2f}%")
        print(f"  Annualized Vol:      {result.metrics['annualized_vol'] * 100:.2f}%")
        print(f"  Diversification:     {result.metrics['diversification_ratio']:.2f}")
        print(f"  Effective N Assets:  {result.metrics['effective_n_assets']:.2f}")
        print(f"  # Rebalances:        {result.metrics['n_rebalances']}")

        print("\nCurrent Weights:")
        print("-" * 40)
        for symbol, weight in zip(symbols, result.weights):
            print(f"  {symbol}: {weight * 100:.2f}%")

        print("\nCorrelation Matrix:")
        print("-" * 40)
        print(result.correlation_matrix.to_string())

        print("\nIndividual Symbol Performance:")
        print("-" * 40)
        symbol_stats = portfolio.get_symbol_statistics(result.symbol_results)
        print(symbol_stats.to_string(index=False))

    print("\n" + "=" * 80)
    print("Portfolio Comparison Summary")
    print("=" * 80)

    # Run all methods again to collect comparative results
    comparison_results = []

    for method in weighting_methods:
        portfolio_config = PortfolioConfig(
            weighting_method=method,
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
            strategy_func=simple_trend_strategy,
            strategy_params=strategy_params,
        )

        comparison_results.append({
            "Method": method,
            "Total Return (%)": result.metrics["total_return"] * 100,
            "Sharpe Ratio": result.metrics["sharpe_ratio"],
            "Max DD (%)": result.metrics["max_drawdown_pct"] * 100,
            "Diversification": result.metrics["diversification_ratio"],
            "Effective N": result.metrics["effective_n_assets"],
        })

    comparison_df = pd.DataFrame(comparison_results)
    print("\n")
    print(comparison_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Example Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
