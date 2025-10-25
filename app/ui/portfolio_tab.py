from datetime import UTC, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.analysis.fourier import smooth_price_series
from core.analysis.signals import generate_signals_with_stops
from core.analysis.stops import compute_atr_stops
from core.backtest.engine import BacktestConfig
from core.data.loader import SUPPORTED_SYMBOLS, load_klines
from core.portfolio.analytics import compute_concentration_metrics
from core.portfolio.portfolio import Portfolio, PortfolioConfig


def render_portfolio_tab() -> None:
    """Render the portfolio management tab."""
    st.header("📊 Portfolio Management")

    st.markdown(
        """
        Build and backtest multi-symbol portfolios with different weighting schemes.
        Compare risk-adjusted performance and analyze correlations across assets.
        """
    )

    # Portfolio Configuration
    st.subheader("Portfolio Configuration")

    col1, col2 = st.columns(2)

    with col1:
        # Symbol selection
        selected_symbols = st.multiselect(
            "Select Symbols",
            options=list(SUPPORTED_SYMBOLS),
            default=["BTCUSDT", "ETHUSDT"],
            help="Choose 2+ symbols for portfolio",
        )

        # Date range
        start_date = st.date_input(
            "Start Date",
            value=datetime(2024, 1, 1, tzinfo=UTC),
            key="portfolio_start_date",
        )
        end_date = st.date_input(
            "End Date",
            value=datetime(2024, 10, 1, tzinfo=UTC),
            key="portfolio_end_date",
        )

        interval = st.selectbox(
            "Interval",
            options=["30m", "1h", "4h"],
            index=1,
            key="portfolio_interval",
        )

    with col2:
        # Weighting method
        weighting_method = st.selectbox(
            "Weighting Method",
            options=["equal", "volatility", "risk_parity"],
            format_func=lambda x: {
                "equal": "Equal Weight",
                "volatility": "Volatility Scaled",
                "risk_parity": "Risk Parity",
            }[x],
            help="Method for determining portfolio weights",
        )

        # Rebalancing
        rebalance_frequency = st.number_input(
            "Rebalance Frequency (hours)",
            min_value=1,
            max_value=720,
            value=24,
            help="Hours between rebalancing",
        )

        # Weight constraints
        max_weight = st.slider(
            "Max Weight per Asset",
            min_value=0.1,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help="Maximum weight for any single asset",
        )

        initial_capital = st.number_input(
            "Initial Capital",
            min_value=1000.0,
            max_value=1000000.0,
            value=10000.0,
            step=1000.0,
        )

    # Strategy Parameters
    with st.expander("⚙️ Strategy Parameters", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            min_trend_hours = st.number_input(
                "Min Trend Period (hours)",
                min_value=1.0,
                max_value=720.0,
                value=48.0,
                step=6.0,
            )

            cutoff_scale = st.slider(
                "Cutoff Scale",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.1,
            )

        with col2:
            atr_period = st.number_input(
                "ATR Period",
                min_value=5,
                max_value=50,
                value=14,
            )

            k_stop = st.slider(
                "Stop Loss Multiplier",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.5,
            )

    # Run Portfolio Backtest
    if st.button("🚀 Run Portfolio Backtest", type="primary"):
        if len(selected_symbols) < 2:
            st.error("Please select at least 2 symbols for portfolio analysis")
            return

        with st.spinner("Loading data and running portfolio backtest..."):
            try:
                # Load data for all symbols
                data_dict = {}
                for symbol in selected_symbols:
                    df = load_klines(
                        symbol=symbol,
                        interval=interval,
                        start=datetime.combine(start_date, datetime.min.time()).replace(tzinfo=UTC),
                        end=datetime.combine(end_date, datetime.min.time()).replace(tzinfo=UTC),
                    )
                    data_dict[symbol] = df

                st.success(f"Loaded data for {len(selected_symbols)} symbols")

                # Define strategy function
                def simple_trend_strategy(df: pd.DataFrame, **params) -> np.ndarray:
                    """Simple trend-following strategy."""
                    close = df["close"].values
                    high = df["high"].values
                    low = df["low"].values

                    # Convert hours to bars
                    interval_hours = {"30m": 0.5, "1h": 1.0, "4h": 4.0}[interval]
                    min_period_bars = int(params["min_trend_hours"] / interval_hours)

                    # Smooth prices
                    smoothed = smooth_price_series(
                        close,
                        min_period_bars=min_period_bars,
                        cutoff_scale=params["cutoff_scale"],
                    )

                    # Compute stops
                    long_stop, _, _, _ = compute_atr_stops(
                        close=close,
                        high=high,
                        low=low,
                        atr_period=params["atr_period"],
                        k_stop=params["k_stop"],
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

                strategy_params = {
                    "min_trend_hours": min_trend_hours,
                    "cutoff_scale": cutoff_scale,
                    "atr_period": atr_period,
                    "k_stop": k_stop,
                }

                # Create portfolio
                portfolio_config = PortfolioConfig(
                    weighting_method=weighting_method,
                    rebalance_frequency=int(rebalance_frequency),
                    rebalance_threshold=0.05,
                    min_weight=0.0,
                    max_weight=max_weight,
                )

                backtest_config = BacktestConfig(
                    initial_capital=initial_capital,
                    fee_rate=0.001,
                    slippage=0.0005,
                )

                portfolio = Portfolio(
                    symbols=selected_symbols,
                    portfolio_config=portfolio_config,
                    backtest_config=backtest_config,
                )

                # Run portfolio backtest
                result = portfolio.run_backtest(
                    data_dict=data_dict,
                    strategy_func=simple_trend_strategy,
                    strategy_params=strategy_params,
                )

                # Store result in session state
                st.session_state["portfolio_result"] = result
                st.session_state["portfolio_symbols"] = selected_symbols

                st.success("Portfolio backtest completed!")

            except Exception as e:
                st.error(f"Error running portfolio backtest: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # Display Results
    if "portfolio_result" in st.session_state:
        result = st.session_state["portfolio_result"]
        symbols = st.session_state["portfolio_symbols"]

        st.divider()
        st.subheader("Portfolio Results")

        # Portfolio Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_return = result.metrics.get("total_return", 0.0)
            st.metric("Total Return", f"{total_return * 100:.2f}%")

        with col2:
            sharpe = result.metrics.get("sharpe_ratio", 0.0)
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")

        with col3:
            max_dd = result.metrics.get("max_drawdown_pct", 0.0)
            st.metric("Max Drawdown", f"{max_dd * 100:.2f}%")

        with col4:
            div_ratio = result.metrics.get("diversification_ratio", 1.0)
            st.metric("Diversification Ratio", f"{div_ratio:.2f}")

        # Additional Metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            effective_n = result.metrics.get("effective_n_assets", len(symbols))
            st.metric("Effective # Assets", f"{effective_n:.1f}")

        with col2:
            n_rebalances = result.metrics.get("n_rebalances", 0)
            st.metric("# Rebalances", n_rebalances)

        with col3:
            max_weight_val = result.metrics.get("max_weight", 0.0)
            st.metric("Max Weight", f"{max_weight_val * 100:.1f}%")

        # Portfolio Equity Curve
        st.subheader("Portfolio Equity Curve")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                y=result.equity_curve,
                mode="lines",
                name="Portfolio",
                line=dict(color="blue", width=2),
            )
        )

        # Mark rebalance dates
        if result.rebalance_dates:
            rebalance_equity = result.equity_curve[result.rebalance_dates]
            fig.add_trace(
                go.Scatter(
                    x=result.rebalance_dates,
                    y=rebalance_equity,
                    mode="markers",
                    name="Rebalances",
                    marker=dict(color="red", size=8, symbol="diamond"),
                )
            )

        fig.update_layout(
            title="Portfolio Equity Over Time",
            xaxis_title="Bar Index",
            yaxis_title="Equity ($)",
            height=500,
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Current Weights
        st.subheader("Current Portfolio Weights")

        weights_df = pd.DataFrame({
            "Symbol": symbols,
            "Weight": result.weights,
            "Weight %": result.weights * 100,
        })

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(
                weights_df.style.format({"Weight": "{:.4f}", "Weight %": "{:.2f}%"}),
                hide_index=True,
            )

        with col2:
            # Pie chart of weights
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=symbols,
                        values=result.weights,
                        hole=0.3,
                    )
                ]
            )
            fig_pie.update_layout(title="Weight Distribution", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Correlation Matrix
        st.subheader("Asset Correlation Matrix")

        if not result.correlation_matrix.empty:
            fig_corr = go.Figure(
                data=go.Heatmap(
                    z=result.correlation_matrix.values,
                    x=result.correlation_matrix.columns,
                    y=result.correlation_matrix.index,
                    colorscale="RdBu",
                    zmid=0,
                    text=result.correlation_matrix.values,
                    texttemplate="%{text:.2f}",
                    textfont={"size": 10},
                    colorbar=dict(title="Correlation"),
                )
            )

            fig_corr.update_layout(
                title="Return Correlations",
                xaxis_title="Symbol",
                yaxis_title="Symbol",
                height=500,
            )

            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Correlation matrix not available")

        # Individual Symbol Performance
        st.subheader("Individual Symbol Performance")

        symbol_stats = portfolio.get_symbol_statistics(result.symbol_results)

        st.dataframe(
            symbol_stats.style.format({
                "total_return": "{:.2%}",
                "sharpe_ratio": "{:.2f}",
                "max_drawdown_pct": "{:.2%}",
                "win_rate": "{:.2%}",
                "profit_factor": "{:.2f}",
            }),
            hide_index=True,
        )

        # Symbol Equity Curves Comparison
        st.subheader("Symbol Equity Curves")

        fig_symbols = go.Figure()

        for symbol in symbols:
            if symbol in result.symbol_results:
                symbol_result = result.symbol_results[symbol]
                # Normalize to percentage
                normalized = (
                    symbol_result.equity_curve / symbol_result.equity_curve[0] - 1
                ) * 100
                fig_symbols.add_trace(
                    go.Scatter(
                        y=normalized,
                        mode="lines",
                        name=symbol,
                    )
                )

        # Add portfolio
        portfolio_normalized = (
            result.equity_curve / result.equity_curve[0] - 1
        ) * 100
        fig_symbols.add_trace(
            go.Scatter(
                y=portfolio_normalized,
                mode="lines",
                name="Portfolio",
                line=dict(color="black", width=3, dash="dash"),
            )
        )

        fig_symbols.update_layout(
            title="Normalized Returns (%)",
            xaxis_title="Bar Index",
            yaxis_title="Return (%)",
            height=500,
            hovermode="x unified",
        )

        st.plotly_chart(fig_symbols, use_container_width=True)

        # Concentration Metrics
        with st.expander("📊 Concentration Metrics", expanded=False):
            concentration = compute_concentration_metrics(result.weights)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Herfindahl Index", f"{concentration['herfindahl_index']:.4f}")
                st.metric("Effective N", f"{concentration['effective_n']:.2f}")

            with col2:
                st.metric("Max Weight", f"{concentration['max_weight'] * 100:.2f}%")
                st.metric("Top 3 Concentration", f"{concentration['top3_concentration'] * 100:.2f}%")
