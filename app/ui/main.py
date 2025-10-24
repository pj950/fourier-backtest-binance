from datetime import UTC, datetime

import plotly.graph_objects as go
import streamlit as st

from core.analysis.fourier import smooth_price_series
from core.analysis.spectral import (
    compute_fft_spectrum,
    compute_sliding_dominant_period,
    create_welch_heatmap,
    find_dominant_peaks,
    plot_fft_spectrum,
    plot_sliding_dominant_period,
)
from core.data.loader import load_klines

st.set_page_config(page_title="Binance Fourier Backtester", layout="wide")

st.title("Binance Fourier Backtester")

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
            st.success(f"Loaded {len(df)} candles")
        except Exception as e:
            st.error(f"Error loading data: {e}")

if "data" in st.session_state and not st.session_state["data"].empty:
    df = st.session_state["data"]

    st.subheader("Fourier Analysis Controls")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        interval_hours = {"30m": 0.5, "1h": 1.0, "4h": 4.0}
        hours_per_bar = interval_hours.get(interval, 1.0)
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
        window_length = st.slider(
            "Window Length (bars)",
            min_value=64,
            max_value=512,
            value=256,
            step=64,
        )

    with col4:
        overlap_pct = st.slider(
            "Window Overlap (%)",
            min_value=0,
            max_value=75,
            value=50,
            step=5,
        )
        overlap_ratio = overlap_pct / 100.0

    show_smoothing = st.checkbox("Show DCT Smoothing", value=True)
    show_spectrum = st.checkbox("Show FFT Spectrum", value=True)
    show_sliding_period = st.checkbox("Show Sliding Window Dominant Period", value=True)
    show_welch_heatmap = st.checkbox("Show Welch PSD Heatmap", value=False)

    st.subheader(f"{symbol} - {interval} OHLCV Chart")

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["open_time"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Price",
            )
        ]
    )

    if show_smoothing:
        with st.spinner("Computing DCT smoothing..."):
            close_prices = df["close"].values
            smoothed_close = smooth_price_series(
                close_prices,
                min_period_bars=min_trend_bars,
                cutoff_scale=cutoff_scale,
            )

            fig.add_trace(
                go.Scatter(
                    x=df["open_time"],
                    y=smoothed_close,
                    mode="lines",
                    name="DCT Smoothed",
                    line=dict(color="orange", width=2),
                )
            )

    fig.add_trace(
        go.Bar(
            x=df["open_time"],
            y=df["volume"],
            name="Volume",
            marker_color="rgba(100, 100, 200, 0.3)",
            yaxis="y2",
        )
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        yaxis2=dict(title="Volume", overlaying="y", side="right"),
        height=600,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )

    st.plotly_chart(fig, use_container_width=True)

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

    st.subheader("Data Summary")
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
