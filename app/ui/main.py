from datetime import UTC, datetime
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.analysis.fourier import smooth_price_series
from core.analysis.signals import generate_signals_with_stops
from core.analysis.spectral import (
    compute_fft_spectrum,
    compute_sliding_dominant_period,
    create_welch_heatmap,
    find_dominant_peaks,
    plot_fft_spectrum,
    plot_sliding_dominant_period,
)
from core.analysis.stops import compute_atr_stops, compute_residual_stops
from core.backtest.engine import BacktestConfig, run_backtest, trades_to_dataframe
from core.data.loader import load_klines

st.set_page_config(page_title="Binance Fourier Backtester", layout="wide")

st.title("Binance Fourier Backtester")

st.markdown("---")
st.subheader("📊 Data Configuration")

col1, col2, col3, col4 = st.columns(4)

with col1:
    symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT"], index=0)

with col2:
    interval = st.selectbox("Interval", ["30m", "1h", "4h"], index=1)

with col3:
    start_date = st.date_input("Start Date", value=datetime(2024, 1, 1).date())

with col4:
    end_date = st.date_input("End Date", value=datetime.now().date())

force_refresh = st.checkbox("Force Refresh (bypass cache)", value=False)

if st.button("Load Data"):
    with st.spinner("Loading data..."):
        try:
            start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=UTC)
            end_dt = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=UTC)

            df = load_klines(
                symbol=symbol,
                interval=interval,
                start=start_dt,
                end=end_dt,
                force_refresh=force_refresh,
            )

            st.session_state["data"] = df
            st.session_state["backtest_result"] = None
            st.success(f"Loaded {len(df)} candles")
        except Exception as e:
            st.error(f"Error loading data: {e}")

if "data" in st.session_state and not st.session_state["data"].empty:
    df = st.session_state["data"]

    st.markdown("---")
    st.subheader("⚙️ Backtest Configuration")

    col1, col2, col3, col4 = st.columns(4)

    interval_hours = {"30m": 0.5, "1h": 1.0, "4h": 4.0}
    hours_per_bar = interval_hours.get(interval, 1.0)

    with col1:
        min_trend_hours = st.number_input(
            "Min Trend Period (hours)",
            min_value=1.0,
            max_value=720.0,
            value=24.0,
            step=1.0,
        )
        min_trend_bars = int(min_trend_hours / hours_per_bar)

    with col2:
        cutoff_scale = st.slider(
            "Cutoff Scale",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Higher = more aggressive smoothing",
        )

    with col3:
        stop_type = st.selectbox("Stop Type", ["ATR", "Residual"], index=0)

    with col4:
        if stop_type == "ATR":
            atr_period = st.number_input("ATR Period", min_value=5, max_value=50, value=14, step=1)
        else:
            residual_window = st.number_input(
                "Residual Window", min_value=5, max_value=100, value=20, step=5
            )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        k_stop = st.slider(
            "Stop Loss Multiplier",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
        )

    with col2:
        k_profit = st.slider(
            "Take Profit Multiplier",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
        )

    with col3:
        slope_threshold = st.number_input(
            "Slope Threshold",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            help="Minimum slope for entry",
        )

    with col4:
        slope_lookback = st.number_input(
            "Slope Lookback (bars)",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        initial_capital = st.number_input(
            "Initial Capital",
            min_value=1000.0,
            max_value=1000000.0,
            value=10000.0,
            step=1000.0,
        )

    with col2:
        fee_rate = st.number_input(
            "Fee Rate (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
        ) / 100.0

    with col3:
        slippage = st.number_input(
            "Slippage (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
        ) / 100.0

    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                close = df["close"].values
                high = df["high"].values
                low = df["low"].values
                open_prices = df["open"].values
                timestamps = df["open_time"]

                smoothed = smooth_price_series(
                    close,
                    min_period_bars=min_trend_bars,
                    cutoff_scale=cutoff_scale,
                )

                if stop_type == "ATR":
                    long_stop, long_profit, short_stop, short_profit = compute_atr_stops(
                        close=close,
                        high=high,
                        low=low,
                        atr_period=atr_period,
                        k_stop=k_stop,
                        k_profit=k_profit,
                    )
                else:
                    long_stop, long_profit, short_stop, short_profit = compute_residual_stops(
                        close=close,
                        smoothed=smoothed,
                        method="sigma",
                        window=residual_window,
                        k_stop=k_stop,
                        k_profit=k_profit,
                    )

                signals = generate_signals_with_stops(
                    close=close,
                    smoothed=smoothed,
                    stop_levels=long_stop,
                    slope_threshold=slope_threshold,
                    slope_lookback=slope_lookback,
                )

                config = BacktestConfig(
                    initial_capital=initial_capital,
                    fee_rate=fee_rate,
                    slippage=slippage,
                    position_size_mode="full",
                    position_size_fraction=1.0,
                )

                result = run_backtest(
                    signals=signals,
                    open_prices=open_prices,
                    high_prices=high,
                    low_prices=low,
                    close_prices=close,
                    timestamps=timestamps,
                    config=config,
                )

                st.session_state["backtest_result"] = result
                st.session_state["smoothed"] = smoothed
                st.session_state["long_stop"] = long_stop
                st.session_state["long_profit"] = long_profit
                st.session_state["signals"] = signals
                st.session_state["config"] = config

                st.success(f"Backtest complete! {result.metrics['n_trades']} trades executed.")

            except Exception as e:
                st.error(f"Error running backtest: {e}")
                import traceback
                st.code(traceback.format_exc())

    if "backtest_result" in st.session_state and st.session_state["backtest_result"] is not None:
        result = st.session_state["backtest_result"]
        smoothed = st.session_state["smoothed"]
        long_stop = st.session_state["long_stop"]
        long_profit = st.session_state["long_profit"]
        signals = st.session_state["signals"]
        config = st.session_state["config"]

        st.markdown("---")
        st.subheader("📈 Performance Summary")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Total Return",
                f"{result.metrics['total_return']:.2%}",
                delta=f"{result.metrics['total_return']:.2%}",
            )

        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{result.metrics['sharpe_ratio']:.2f}",
            )

        with col3:
            st.metric(
                "Max Drawdown",
                f"{result.metrics['max_drawdown_pct']:.2%}",
                delta=f"{result.metrics['max_drawdown_pct']:.2%}",
                delta_color="inverse",
            )

        with col4:
            st.metric(
                "Win Rate",
                f"{result.metrics['win_rate']:.2%}",
            )

        with col5:
            st.metric(
                "# Trades",
                f"{result.metrics['n_trades']:.0f}",
            )

        with st.expander("📊 Detailed Metrics", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Returns & Risk**")
                metrics_df1 = pd.DataFrame(
                    {
                        "Metric": [
                            "Cumulative Return",
                            "Annualized Return",
                            "Max Drawdown ($)",
                            "Max Drawdown (%)",
                            "Sharpe Ratio",
                            "Sortino Ratio",
                        ],
                        "Value": [
                            f"{result.metrics['cumulative_return']:.2%}",
                            f"{result.metrics['annualized_return']:.2%}",
                            f"${result.metrics['max_drawdown']:.2f}",
                            f"{result.metrics['max_drawdown_pct']:.2%}",
                            f"{result.metrics['sharpe_ratio']:.2f}",
                            f"{result.metrics['sortino_ratio']:.2f}",
                        ],
                    }
                )
                st.dataframe(metrics_df1, use_container_width=True, hide_index=True)

            with col2:
                st.markdown("**Trade Statistics**")
                metrics_df2 = pd.DataFrame(
                    {
                        "Metric": [
                            "Number of Trades",
                            "Wins / Losses",
                            "Win Rate",
                            "Profit Factor",
                            "Avg Win",
                            "Avg Loss",
                        ],
                        "Value": [
                            f"{result.metrics['n_trades']:.0f}",
                            f"{result.metrics['n_wins']:.0f} / {result.metrics['n_losses']:.0f}",
                            f"{result.metrics['win_rate']:.2%}",
                            f"{result.metrics['profit_factor']:.2f}",
                            f"${result.metrics['avg_win']:.2f}",
                            f"${result.metrics['avg_loss']:.2f}",
                        ],
                    }
                )
                st.dataframe(metrics_df2, use_container_width=True, hide_index=True)

        with st.expander("🎛️ Parameter Snapshot", expanded=False):
            params_df = pd.DataFrame(
                {
                    "Parameter": [
                        "Symbol",
                        "Interval",
                        "Min Trend Period",
                        "Cutoff Scale",
                        "Stop Type",
                        "Stop Loss Multiplier",
                        "Take Profit Multiplier",
                        "Slope Threshold",
                        "Initial Capital",
                        "Fee Rate",
                        "Slippage",
                    ],
                    "Value": [
                        symbol,
                        interval,
                        f"{min_trend_hours}h ({min_trend_bars} bars)",
                        f"{cutoff_scale:.1f}",
                        f"{stop_type} ({atr_period if stop_type == 'ATR' else residual_window})",
                        f"{k_stop:.1f}",
                        f"{k_profit:.1f}",
                        f"{slope_threshold:.2f}",
                        f"${initial_capital:.2f}",
                        f"{fee_rate*100:.3f}%",
                        f"{slippage*100:.3f}%",
                    ],
                }
            )
            st.dataframe(params_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("📉 Price Chart with Entry/Exit Markers")

        entry_indices = np.where(signals == 1)[0]
        exit_indices = np.where(signals == -1)[0]

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=df["open_time"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Price",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["open_time"],
                y=smoothed,
                mode="lines",
                name="Smoothed Trend",
                line=dict(color="orange", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["open_time"],
                y=long_stop,
                mode="lines",
                name="Stop Loss",
                line=dict(color="red", width=1, dash="dash"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["open_time"],
                y=long_profit,
                mode="lines",
                name="Take Profit",
                line=dict(color="green", width=1, dash="dash"),
            )
        )

        if len(entry_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=df.iloc[entry_indices]["open_time"],
                    y=df.iloc[entry_indices]["close"],
                    mode="markers",
                    name="Entry",
                    marker=dict(color="lime", size=10, symbol="triangle-up"),
                )
            )

        if len(exit_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=df.iloc[exit_indices]["open_time"],
                    y=df.iloc[exit_indices]["close"],
                    mode="markers",
                    name="Exit",
                    marker=dict(color="red", size=10, symbol="triangle-down"),
                )
            )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Price (USDT)",
            height=600,
            hovermode="x unified",
            xaxis_rangeslider_visible=False,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("💰 Equity Curve with Drawdown")

        equity = result.equity_curve
        cummax = np.maximum.accumulate(equity)
        drawdown = equity - cummax
        drawdown_pct = drawdown / cummax * 100

        fig_equity = go.Figure()

        fig_equity.add_trace(
            go.Scatter(
                x=df["open_time"],
                y=equity,
                mode="lines",
                name="Equity",
                line=dict(color="blue", width=2),
            )
        )

        fig_equity.add_trace(
            go.Scatter(
                x=df["open_time"],
                y=cummax,
                mode="lines",
                name="High Water Mark",
                line=dict(color="lightblue", width=1, dash="dash"),
            )
        )

        drawdown_mask = drawdown < 0
        if np.any(drawdown_mask):
            fig_equity.add_trace(
                go.Scatter(
                    x=df["open_time"],
                    y=equity,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(255, 0, 0, 0.2)",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            fig_equity.add_trace(
                go.Scatter(
                    x=df["open_time"],
                    y=cummax,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        fig_equity.update_layout(
            xaxis_title="Time",
            yaxis_title="Equity ($)",
            height=500,
            hovermode="x unified",
        )

        st.plotly_chart(fig_equity, use_container_width=True)

        fig_dd = go.Figure()

        fig_dd.add_trace(
            go.Scatter(
                x=df["open_time"],
                y=drawdown_pct,
                mode="lines",
                name="Drawdown %",
                line=dict(color="red", width=2),
                fill="tozeroy",
                fillcolor="rgba(255, 0, 0, 0.3)",
            )
        )

        fig_dd.update_layout(
            xaxis_title="Time",
            yaxis_title="Drawdown (%)",
            height=300,
            hovermode="x unified",
        )

        st.plotly_chart(fig_dd, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Trade Log")

        if len(result.trades) > 0:
            trades_df = trades_to_dataframe(result.trades)

            col1, col2 = st.columns(2)

            with col1:
                trade_filter = st.selectbox(
                    "Filter by Result",
                    ["All", "Winning", "Losing"],
                    index=0,
                )

            with col2:
                sort_by = st.selectbox(
                    "Sort by",
                    ["Entry Time", "PnL", "PnL %", "Bars Held"],
                    index=0,
                )

            filtered_trades = trades_df.copy()

            if trade_filter == "Winning":
                filtered_trades = filtered_trades[filtered_trades["pnl"] > 0]
            elif trade_filter == "Losing":
                filtered_trades = filtered_trades[filtered_trades["pnl"] <= 0]

            sort_column_map = {
                "Entry Time": "entry_time",
                "PnL": "pnl",
                "PnL %": "pnl_pct",
                "Bars Held": "bars_held",
            }
            filtered_trades = filtered_trades.sort_values(
                by=sort_column_map[sort_by], ascending=False
            )

            display_trades = filtered_trades[
                [
                    "entry_time",
                    "exit_time",
                    "entry_price",
                    "exit_price",
                    "pnl",
                    "pnl_pct",
                    "bars_held",
                    "mae",
                    "mfe",
                ]
            ].copy()

            display_trades["entry_time"] = pd.to_datetime(display_trades["entry_time"]).dt.strftime(
                "%Y-%m-%d %H:%M"
            )
            display_trades["exit_time"] = pd.to_datetime(display_trades["exit_time"]).dt.strftime(
                "%Y-%m-%d %H:%M"
            )
            display_trades["pnl"] = display_trades["pnl"].apply(lambda x: f"${x:.2f}")
            display_trades["pnl_pct"] = display_trades["pnl_pct"].apply(lambda x: f"{x*100:.2f}%")
            display_trades["entry_price"] = display_trades["entry_price"].apply(
                lambda x: f"${x:.2f}"
            )
            display_trades["exit_price"] = display_trades["exit_price"].apply(lambda x: f"${x:.2f}")
            display_trades["mae"] = display_trades["mae"].apply(lambda x: f"${x:.2f}")
            display_trades["mfe"] = display_trades["mfe"].apply(lambda x: f"${x:.2f}")

            display_trades.columns = [
                "Entry Time",
                "Exit Time",
                "Entry Price",
                "Exit Price",
                "PnL",
                "PnL %",
                "Bars Held",
                "MAE",
                "MFE",
            ]

            st.dataframe(display_trades, use_container_width=True, hide_index=True)

            csv_buffer = StringIO()
            trades_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="📥 Download Trade Log (CSV)",
                data=csv_data,
                file_name=f"trades_{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        else:
            st.info("No trades executed in this backtest.")

    st.markdown("---")
    st.subheader("🔬 Fourier Analysis")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        window_length = st.slider(
            "Window Length (bars)",
            min_value=64,
            max_value=512,
            value=256,
            step=64,
        )

    with col2:
        overlap_pct = st.slider(
            "Window Overlap (%)",
            min_value=0,
            max_value=75,
            value=50,
            step=5,
        )
        overlap_ratio = overlap_pct / 100.0

    with col3:
        show_spectrum = st.checkbox("Show FFT Spectrum", value=True)

    with col4:
        show_sliding_period = st.checkbox("Show Sliding Dominant Period", value=True)

    show_welch_heatmap = st.checkbox("Show Welch PSD Heatmap", value=False)

    if show_spectrum:
        st.subheader("FFT Power Spectrum")
        with st.spinner("Computing FFT spectrum..."):
            close_prices = df["close"].values
            frequencies, power_spectrum = compute_fft_spectrum(close_prices)
            peaks = find_dominant_peaks(frequencies, power_spectrum, n_peaks=5)

            spectrum_fig = plot_fft_spectrum(frequencies, power_spectrum, peaks, interval)
            st.plotly_chart(spectrum_fig, use_container_width=True)

            if peaks:
                st.write("**Dominant Peaks:**")
                cols = st.columns(min(len(peaks), 5))
                for i, (col, peak) in enumerate(zip(cols, peaks)):
                    period_bars = peak["period"]
                    period_hours = period_bars * hours_per_bar
                    with col:
                        st.metric(
                            f"Peak {i + 1}",
                            f"{period_bars:.1f} bars",
                            f"{period_hours:.1f}h",
                        )

    if show_sliding_period:
        st.subheader("Sliding Window Dominant Period")
        with st.spinner("Computing sliding window analysis..."):
            close_prices = df["close"].values
            time_indices, dominant_periods = compute_sliding_dominant_period(
                close_prices,
                window_length=window_length,
                overlap_ratio=overlap_ratio,
            )

            sliding_fig = plot_sliding_dominant_period(
                df["open_time"],
                time_indices,
                dominant_periods,
                interval,
            )
            st.plotly_chart(sliding_fig, use_container_width=True)

    if show_welch_heatmap:
        st.subheader("Welch PSD Heatmap")
        with st.spinner("Computing Welch PSD heatmap..."):
            close_prices = df["close"].values
            heatmap_fig = create_welch_heatmap(
                close_prices,
                df["open_time"],
                window_length=window_length,
                overlap_ratio=overlap_ratio,
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Data Summary")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Rows", len(df))

    with col2:
        st.metric("First Date", df["open_time"].min().strftime("%Y-%m-%d"))

    with col3:
        st.metric("Last Date", df["open_time"].max().strftime("%Y-%m-%d"))

    with col4:
        st.metric("Avg Volume", f"{df['volume'].mean():.2f}")

    with col5:
        st.metric("Price Range", f"${df['low'].min():.2f} - ${df['high'].max():.2f}")

    with st.expander("View Raw Data"):
        st.dataframe(df, use_container_width=True)
else:
    st.info("👆 Select parameters and click 'Load Data' to get started")
