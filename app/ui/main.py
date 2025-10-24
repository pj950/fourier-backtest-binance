from datetime import UTC, datetime

import plotly.graph_objects as go
import streamlit as st

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
