"""Complete example of parameter optimization and robustness evaluation."""

import numpy as np
import pandas as pd

from core.analysis.fourier import smooth_price_series
from core.analysis.signals import generate_signals_with_stops
from core.analysis.stops import compute_atr_stops
from core.backtest.engine import BacktestConfig, run_backtest
from core.data.loader import load_klines
from core.optimization.export import (
    export_full_optimization_results,
    export_monte_carlo_results,
    export_walkforward_results,
)
from core.optimization.monte_carlo import compute_mc_metrics, monte_carlo_equity_curves
from core.optimization.params import StrategyParams, create_default_param_space
from core.optimization.runner import OptimizationRunner
from core.optimization.visualization import (
    plot_frontier,
    plot_monte_carlo_distribution,
    plot_param_heatmap,
    plot_walkforward_results,
)


def strategy_objective_function(params: StrategyParams, df: pd.DataFrame) -> dict[str, float]:
    """
    Objective function that runs backtest with given parameters.

    Args:
        params: Strategy parameters
        df: OHLCV dataframe

    Returns:
        Dictionary of performance metrics
    """
    if len(df) < 100:
        return {"sharpe_ratio": -np.inf, "total_return": -1.0}

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_prices = df["open"].values
    timestamps = df["open_time"]

    interval_hours = 1.0
    min_period_bars = int(params.min_trend_period_hours / interval_hours)

    try:
        smoothed = smooth_price_series(
            close,
            min_period_bars=min_period_bars,
            cutoff_scale=params.cutoff_scale,
        )

        long_stop, long_profit, _, _ = compute_atr_stops(
            close,
            high,
            low,
            atr_period=params.atr_period,
            k_stop=params.k_stop,
            k_profit=params.k_profit,
        )

        signals = generate_signals_with_stops(
            close=close,
            smoothed=smoothed,
            stop_levels=long_stop,
            slope_threshold=params.slope_threshold,
            slope_lookback=params.slope_lookback,
            min_volatility=params.min_volatility,
        )

        config = BacktestConfig(
            initial_capital=params.initial_capital,
            fee_rate=params.fee_rate,
            slippage=params.slippage,
        )

        result = run_backtest(signals, open_prices, high, low, close, timestamps, config)

        return result.metrics

    except Exception as e:
        print(f"Error in backtest: {e}")
        return {"sharpe_ratio": -np.inf, "total_return": -1.0}


def main():
    """Run complete optimization example."""
    print("=" * 80)
    print("M8: Parameter Optimization and Robustness Evaluation")
    print("=" * 80)

    from datetime import datetime, timedelta

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=180)

    print("\n1. Loading data...")
    df = load_klines("BTCUSDT", "1h", start_date, end_date)
    print(f"Loaded {len(df)} bars from {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}")

    param_spaces = create_default_param_space()

    runner = OptimizationRunner(
        objective_function=strategy_objective_function,
        objective_metric="sharpe_ratio",
        maximize=True,
        seed=42,
    )

    print("\n2. Running Random Search (100 iterations)...")
    random_run = runner.run_random_search(
        param_spaces=param_spaces,
        data=df,
        n_iter=100,
        verbose=True,
    )

    print(f"\nRandom Search Results:")
    print(f"  Best Sharpe: {random_run.best_score:.4f}")
    print(f"  Runtime: {random_run.total_runtime:.2f}s")
    print(f"  Best params: {random_run.best_params}")

    print("\n3. Running Bayesian Optimization (10 initial + 40 BO iterations)...")
    bayesian_run = runner.run_bayesian_search(
        param_spaces=param_spaces,
        data=df,
        n_initial=10,
        n_iter=40,
        verbose=True,
    )

    print(f"\nBayesian Optimization Results:")
    print(f"  Best Sharpe: {bayesian_run.best_score:.4f}")
    print(f"  Runtime: {bayesian_run.total_runtime:.2f}s")
    print(f"  Best params: {bayesian_run.best_params}")

    print("\n4. Running Walk-Forward Analysis...")
    train_size = len(df) // 3
    test_size = len(df) // 6
    step_size = test_size

    wf_run, wf_result = runner.run_walkforward(
        param_spaces=param_spaces,
        data=df,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        anchored=False,
        search_method="random",
        n_candidates=30,
        verbose=True,
    )

    print(f"\nWalk-Forward Results:")
    print(f"  Number of windows: {len(wf_result.windows)}")
    print(f"  Combined OOS Sharpe: {wf_result.combined_oos_metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  Combined OOS Return: {wf_result.combined_oos_metrics.get('total_return', 0):.4f}")

    print("\n5. Running Monte Carlo Analysis...")
    best_metrics = strategy_objective_function(bayesian_run.best_params, df)
    config = BacktestConfig(
        initial_capital=bayesian_run.best_params.initial_capital,
        fee_rate=bayesian_run.best_params.fee_rate,
        slippage=bayesian_run.best_params.slippage,
    )

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_prices = df["open"].values
    timestamps = df["open_time"]

    min_period_bars = int(bayesian_run.best_params.min_trend_period_hours / 1.0)
    smoothed = smooth_price_series(close, min_period_bars, bayesian_run.best_params.cutoff_scale)
    long_stop, _, _, _ = compute_atr_stops(
        close, high, low, bayesian_run.best_params.atr_period,
        bayesian_run.best_params.k_stop, bayesian_run.best_params.k_profit,
    )
    signals = generate_signals_with_stops(
        close, smoothed, long_stop, bayesian_run.best_params.slope_threshold,
        bayesian_run.best_params.slope_lookback,
    )
    result = run_backtest(signals, open_prices, high, low, close, timestamps, config)

    mc_equity_curves = monte_carlo_equity_curves(
        result.equity_curve,
        n_simulations=1000,
        block_size=24,
        seed=42,
    )

    mc_result = compute_mc_metrics(mc_equity_curves, config.initial_capital)

    print(f"\nMonte Carlo Results (1000 simulations):")
    print(f"  Mean Sharpe: {mc_result.mean_metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  Std Sharpe: {mc_result.std_metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  Sharpe 5th percentile: {mc_result.percentiles['sharpe_ratio']['p5']:.4f}")
    print(f"  Sharpe 95th percentile: {mc_result.percentiles['sharpe_ratio']['p95']:.4f}")

    print("\n6. Exporting results...")
    export_full_optimization_results(bayesian_run, "optimization_results/bayesian", include_visualizations=True)
    export_walkforward_results(wf_result, "optimization_results/walkforward")
    export_monte_carlo_results(mc_result, "optimization_results/monte_carlo")

    print("\n7. Generating visualizations...")
    try:
        fig = plot_param_heatmap(
            bayesian_run.leaderboard,
            "k_stop",
            "k_profit",
            "train_sharpe_ratio",
        )
        fig.savefig("optimization_results/heatmap_stops.png", dpi=150, bbox_inches="tight")
        print("  Saved heatmap_stops.png")

        fig = plot_frontier(
            bayesian_run.leaderboard,
            "train_sharpe_ratio",
            "train_max_drawdown_pct",
        )
        fig.savefig("optimization_results/frontier.png", dpi=150, bbox_inches="tight")
        print("  Saved frontier.png")

        fig = plot_walkforward_results(wf_result, "sharpe_ratio")
        fig.savefig("optimization_results/walkforward_sharpe.png", dpi=150, bbox_inches="tight")
        print("  Saved walkforward_sharpe.png")

        fig = plot_monte_carlo_distribution(mc_result, "sharpe_ratio")
        fig.savefig("optimization_results/mc_sharpe_distribution.png", dpi=150, bbox_inches="tight")
        print("  Saved mc_sharpe_distribution.png")

    except Exception as e:
        print(f"  Warning: Could not generate all visualizations: {e}")

    print("\n" + "=" * 80)
    print("Optimization complete! Results saved to optimization_results/")
    print("=" * 80)


if __name__ == "__main__":
    main()
