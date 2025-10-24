"""Streamlit UI tab for parameter optimization."""

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from core.data.loader import load_klines
from core.optimization.export import (
    export_full_optimization_results,
    export_monte_carlo_results,
    export_walkforward_results,
)
from core.optimization.monte_carlo import compute_mc_metrics, monte_carlo_equity_curves
from core.optimization.params import create_default_param_space
from core.optimization.runner import OptimizationRunner
from core.optimization.visualization import (
    plot_frontier,
    plot_monte_carlo_distribution,
    plot_optimization_progress,
    plot_param_heatmap,
    plot_parameter_importance,
    plot_walkforward_results,
)


def _create_objective_function():
    """Create objective function for optimization."""
    import numpy as np
    from core.analysis.fourier import smooth_price_series
    from core.analysis.signals import generate_signals_with_stops
    from core.analysis.stops import compute_atr_stops
    from core.backtest.engine import BacktestConfig, run_backtest
    from core.optimization.params import StrategyParams

    def objective_function(params: StrategyParams, df: pd.DataFrame) -> dict[str, float]:
        if len(df) < 100:
            return {"sharpe_ratio": -np.inf}

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        open_prices = df["open"].values
        timestamps = df["open_time"]

        interval_map = {"30m": 0.5, "1h": 1.0, "4h": 4.0}
        interval_hours = interval_map.get(st.session_state.get("opt_interval", "1h"), 1.0)
        min_period_bars = int(params.min_trend_period_hours / interval_hours)

        try:
            smoothed = smooth_price_series(close, min_period_bars, params.cutoff_scale)
            long_stop, _, _, _ = compute_atr_stops(
                close, high, low, params.atr_period, params.k_stop, params.k_profit
            )
            signals = generate_signals_with_stops(
                close, smoothed, long_stop, params.slope_threshold, params.slope_lookback
            )
            config = BacktestConfig(
                initial_capital=params.initial_capital,
                fee_rate=params.fee_rate,
                slippage=params.slippage,
            )
            result = run_backtest(signals, open_prices, high, low, close, timestamps, config)
            return result.metrics
        except Exception:
            return {"sharpe_ratio": -np.inf}

    return objective_function


def render_optimization_tab():
    """Render the optimization tab in Streamlit UI."""
    st.header("🔍 Parameter Optimization & Robustness")

    st.markdown("""
    Optimize strategy parameters using Grid Search, Random Search, or Bayesian Optimization.
    Evaluate robustness with Walk-Forward Analysis and Monte Carlo resampling.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Configuration")
        symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT"], key="opt_symbol")
        interval = st.selectbox("Interval", ["30m", "1h", "4h"], key="opt_interval")

        days_back = st.slider("Days of data", 30, 365, 180, key="opt_days_back")
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)

    with col2:
        st.subheader("Optimization Settings")
        search_method = st.selectbox(
            "Search Method",
            ["Random Search", "Grid Search", "Bayesian Optimization"],
            key="opt_search_method",
        )

        objective_metric = st.selectbox(
            "Objective Metric",
            ["sharpe_ratio", "sortino_ratio", "total_return", "profit_factor"],
            key="opt_objective_metric",
        )

        seed = st.number_input("Random Seed", value=42, key="opt_seed")

    with st.expander("📊 Search Parameters", expanded=True):
        if search_method == "Random Search":
            n_iter = st.slider("Number of iterations", 10, 500, 100, key="opt_random_n_iter")
        elif search_method == "Grid Search":
            n_points = st.slider("Points per parameter", 2, 10, 3, key="opt_grid_points")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                n_initial = st.slider("Initial random samples", 5, 50, 10, key="opt_bo_initial")
            with col_b:
                n_iter_bo = st.slider("BO iterations", 10, 100, 40, key="opt_bo_iter")

    with st.expander("🔄 Walk-Forward Analysis", expanded=False):
        enable_wf = st.checkbox("Enable Walk-Forward Analysis", value=False, key="opt_enable_wf")
        if enable_wf:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                train_pct = st.slider("Train %", 30, 70, 50, key="opt_wf_train_pct")
            with col_b:
                test_pct = st.slider("Test %", 10, 40, 20, key="opt_wf_test_pct")
            with col_c:
                wf_anchored = st.checkbox("Anchored", value=False, key="opt_wf_anchored")

            wf_candidates = st.slider(
                "Candidates per window", 10, 100, 30, key="opt_wf_candidates"
            )

    with st.expander("🎲 Monte Carlo Resampling", expanded=False):
        enable_mc = st.checkbox("Enable Monte Carlo Analysis", value=False, key="opt_enable_mc")
        if enable_mc:
            col_a, col_b = st.columns(2)
            with col_a:
                n_simulations = st.slider("Simulations", 100, 5000, 1000, key="opt_mc_sims")
            with col_b:
                block_size = st.slider("Block size", 5, 100, 24, key="opt_mc_block")

    if "opt_running" not in st.session_state:
        st.session_state.opt_running = False

    if "opt_results" not in st.session_state:
        st.session_state.opt_results = None

    col_run, col_export = st.columns([1, 1])

    with col_run:
        if st.button("🚀 Run Optimization", disabled=st.session_state.opt_running, use_container_width=True):
            st.session_state.opt_running = True
            st.rerun()

    with col_export:
        export_enabled = st.session_state.opt_results is not None
        if st.button("💾 Export Results", disabled=not export_enabled, use_container_width=True):
            if st.session_state.opt_results:
                _export_results()

    if st.session_state.opt_running:
        _run_optimization_workflow(
            symbol, interval, start_date, end_date, search_method,
            objective_metric, seed, n_iter if search_method == "Random Search" else None,
            n_points if search_method == "Grid Search" else None,
            n_initial if search_method == "Bayesian Optimization" else None,
            n_iter_bo if search_method == "Bayesian Optimization" else None,
            enable_wf, train_pct if enable_wf else None, test_pct if enable_wf else None,
            wf_anchored if enable_wf else None, wf_candidates if enable_wf else None,
            enable_mc, n_simulations if enable_mc else None, block_size if enable_mc else None,
        )

    if st.session_state.opt_results:
        _display_results()


def _run_optimization_workflow(
    symbol, interval, start_date, end_date, search_method, objective_metric, seed,
    n_iter, n_points, n_initial, n_iter_bo, enable_wf, train_pct, test_pct,
    wf_anchored, wf_candidates, enable_mc, n_simulations, block_size,
):
    """Run the optimization workflow."""
    try:
        with st.spinner("Loading data..."):
            df = load_klines(symbol, interval, start_date, end_date)
            st.info(f"Loaded {len(df)} bars")

        param_spaces = create_default_param_space()
        objective_function = _create_objective_function()

        runner = OptimizationRunner(
            objective_function=objective_function,
            objective_metric=objective_metric,
            maximize=True,
            seed=seed,
        )

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        status_text.text(f"Running {search_method}...")
        progress_bar.progress(0.2)

        if search_method == "Random Search":
            opt_run = runner.run_random_search(param_spaces, df, n_iter, verbose=False)
        elif search_method == "Grid Search":
            opt_run = runner.run_grid_search(param_spaces, df, n_points, verbose=False)
        else:
            opt_run = runner.run_bayesian_search(param_spaces, df, n_initial, n_iter_bo, verbose=False)

        progress_bar.progress(0.5)

        wf_result = None
        if enable_wf:
            status_text.text("Running Walk-Forward Analysis...")
            n_bars = len(df)
            train_size = int(n_bars * train_pct / 100)
            test_size = int(n_bars * test_pct / 100)

            _, wf_result = runner.run_walkforward(
                param_spaces, df, train_size, test_size, anchored=wf_anchored,
                search_method="random", n_candidates=wf_candidates, verbose=False,
            )

        progress_bar.progress(0.7)

        mc_result = None
        if enable_mc:
            status_text.text("Running Monte Carlo Resampling...")
            from core.analysis.fourier import smooth_price_series
            from core.analysis.signals import generate_signals_with_stops
            from core.analysis.stops import compute_atr_stops
            from core.backtest.engine import BacktestConfig, run_backtest

            close = df["close"].values
            high = df["high"].values
            low = df["low"].values
            open_prices = df["open"].values
            timestamps = df["open_time"]

            interval_map = {"30m": 0.5, "1h": 1.0, "4h": 4.0}
            interval_hours = interval_map.get(interval, 1.0)
            min_period_bars = int(opt_run.best_params.min_trend_period_hours / interval_hours)

            smoothed = smooth_price_series(close, min_period_bars, opt_run.best_params.cutoff_scale)
            long_stop, _, _, _ = compute_atr_stops(
                close, high, low, opt_run.best_params.atr_period,
                opt_run.best_params.k_stop, opt_run.best_params.k_profit,
            )
            signals = generate_signals_with_stops(
                close, smoothed, long_stop, opt_run.best_params.slope_threshold,
                opt_run.best_params.slope_lookback,
            )
            config = BacktestConfig(
                initial_capital=opt_run.best_params.initial_capital,
                fee_rate=opt_run.best_params.fee_rate,
                slippage=opt_run.best_params.slippage,
            )
            result = run_backtest(signals, open_prices, high, low, close, timestamps, config)

            mc_curves = monte_carlo_equity_curves(
                result.equity_curve, n_simulations, block_size, seed,
            )
            mc_result = compute_mc_metrics(mc_curves, config.initial_capital)

        progress_bar.progress(1.0)
        status_text.text("✅ Optimization complete!")

        st.session_state.opt_results = {
            "opt_run": opt_run,
            "wf_result": wf_result,
            "mc_result": mc_result,
        }

        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

    except Exception as e:
        st.error(f"Error during optimization: {e}")
        import traceback
        st.code(traceback.format_exc())

    finally:
        st.session_state.opt_running = False
        st.rerun()


def _display_results():
    """Display optimization results."""
    results = st.session_state.opt_results
    opt_run = results["opt_run"]
    wf_result = results.get("wf_result")
    mc_result = results.get("mc_result")

    st.success(f"✅ Optimization complete! Best {opt_run.objective_metric}: {opt_run.best_score:.4f}")

    tabs = st.tabs(["📊 Leaderboard", "🎯 Best Config", "📈 Visualizations", "🔄 Walk-Forward", "🎲 Monte Carlo"])

    with tabs[0]:
        st.subheader("Leaderboard")
        st.dataframe(opt_run.leaderboard.head(20), use_container_width=True)

        st.download_button(
            label="Download Full Leaderboard CSV",
            data=opt_run.leaderboard.to_csv(index=False),
            file_name="leaderboard.csv",
            mime="text/csv",
        )

    with tabs[1]:
        st.subheader("Best Configuration")
        st.json(opt_run.best_params.to_dict())

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Score", f"{opt_run.best_score:.4f}")
        with col2:
            st.metric("Total Runtime", f"{opt_run.total_runtime:.2f}s")
        with col3:
            st.metric("Candidates", len(opt_run.results))

    with tabs[2]:
        st.subheader("Visualizations")

        try:
            st.pyplot(plot_optimization_progress(opt_run))
        except Exception as e:
            st.warning(f"Could not plot optimization progress: {e}")

        try:
            st.pyplot(plot_parameter_importance(opt_run.leaderboard, f"train_{opt_run.objective_metric}"))
        except Exception as e:
            st.warning(f"Could not plot parameter importance: {e}")

        try:
            st.pyplot(plot_frontier(opt_run.leaderboard, f"train_{opt_run.objective_metric}", "train_max_drawdown_pct"))
        except Exception as e:
            st.warning(f"Could not plot frontier: {e}")

        st.subheader("Parameter Heatmaps")
        col1, col2 = st.columns(2)
        with col1:
            param_x = st.selectbox("X-axis parameter", ["k_stop", "k_profit", "atr_period", "cutoff_scale"])
        with col2:
            param_y = st.selectbox("Y-axis parameter", ["k_profit", "k_stop", "min_trend_period_hours", "atr_period"])

        try:
            st.pyplot(plot_param_heatmap(opt_run.leaderboard, param_x, param_y, f"train_{opt_run.objective_metric}"))
        except Exception as e:
            st.warning(f"Could not plot heatmap: {e}")

    with tabs[3]:
        if wf_result:
            st.subheader("Walk-Forward Analysis")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Windows", len(wf_result.windows))
            with col2:
                oos_sharpe = wf_result.combined_oos_metrics.get("sharpe_ratio", 0)
                st.metric("OOS Sharpe", f"{oos_sharpe:.4f}")
            with col3:
                oos_return = wf_result.combined_oos_metrics.get("total_return", 0)
                st.metric("OOS Return", f"{oos_return:.2%}")

            st.pyplot(plot_walkforward_results(wf_result, "sharpe_ratio"))

            st.subheader("Window Metrics")
            wf_df = pd.DataFrame(wf_result.test_metrics)
            wf_df["window_id"] = wf_df.index
            st.dataframe(wf_df, use_container_width=True)
        else:
            st.info("Walk-Forward Analysis not enabled for this run")

    with tabs[4]:
        if mc_result:
            st.subheader("Monte Carlo Analysis")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Simulations", mc_result.n_simulations)
            with col2:
                mean_sharpe = mc_result.mean_metrics.get("sharpe_ratio", 0)
                st.metric("Mean Sharpe", f"{mean_sharpe:.4f}")
            with col3:
                p5 = mc_result.percentiles["sharpe_ratio"]["p5"]
                st.metric("5th Percentile", f"{p5:.4f}")
            with col4:
                p95 = mc_result.percentiles["sharpe_ratio"]["p95"]
                st.metric("95th Percentile", f"{p95:.4f}")

            st.pyplot(plot_monte_carlo_distribution(mc_result, "sharpe_ratio"))

            st.subheader("Metric Statistics")
            stats_data = []
            for metric in mc_result.mean_metrics.keys():
                stats_data.append({
                    "Metric": metric,
                    "Mean": f"{mc_result.mean_metrics[metric]:.4f}",
                    "Std": f"{mc_result.std_metrics[metric]:.4f}",
                    "P5": f"{mc_result.percentiles[metric]['p5']:.4f}",
                    "P50": f"{mc_result.percentiles[metric]['p50']:.4f}",
                    "P95": f"{mc_result.percentiles[metric]['p95']:.4f}",
                })
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
        else:
            st.info("Monte Carlo Analysis not enabled for this run")


def _export_results():
    """Export optimization results."""
    results = st.session_state.opt_results
    opt_run = results["opt_run"]
    wf_result = results.get("wf_result")
    mc_result = results.get("mc_result")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"optimization_results/{timestamp}")

    try:
        export_full_optimization_results(opt_run, output_dir, include_visualizations=True)

        if wf_result:
            export_walkforward_results(wf_result, output_dir / "walkforward")

        if mc_result:
            export_monte_carlo_results(mc_result, output_dir / "monte_carlo")

        st.success(f"✅ Results exported to {output_dir}")
    except Exception as e:
        st.error(f"Export failed: {e}")
