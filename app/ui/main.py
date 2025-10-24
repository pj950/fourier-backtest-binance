import time
import traceback
from datetime import UTC, datetime
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui.live import LiveComputationConfig, LiveDataCoordinator
from config.presets import PresetError, PresetManager, UIConfig
from config.settings import settings
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
from core.data.exceptions import (
    BinanceRateLimitError,
    BinanceRequestError,
    BinanceTransientError,
)
from core.data.loader import SUPPORTED_INTERVALS, SUPPORTED_SYMBOLS, load_klines
from core.data.streaming import StreamState

preset_manager = PresetManager(settings.preset_storage_path, settings.last_session_state_path)

SYMBOL_OPTIONS = list(SUPPORTED_SYMBOLS)
INTERVAL_OPTIONS = list(SUPPORTED_INTERVALS)
STOP_TYPES = ["ATR", "Residual"]
PRESET_PLACEHOLDER = "— Select Preset —"


def _safe_index(options: list[str], value: str | None) -> int:
    if value is None:
        return 0
    try:
        return options.index(value)
    except ValueError:
        return 0


def _apply_config_to_session(config: UIConfig) -> None:
    st.session_state["symbol"] = config.symbol
    st.session_state["interval"] = config.interval
    st.session_state["start_date"] = config.start_date
    st.session_state["end_date"] = config.end_date
    st.session_state["force_refresh"] = config.force_refresh
    st.session_state["min_trend_hours"] = config.min_trend_hours
    st.session_state["cutoff_scale"] = config.cutoff_scale
    st.session_state["stop_type"] = config.stop_type
    st.session_state["atr_period"] = config.atr_period
    st.session_state["residual_window"] = config.residual_window
    st.session_state["k_stop"] = config.k_stop
    st.session_state["k_profit"] = config.k_profit
    st.session_state["slope_threshold"] = config.slope_threshold
    st.session_state["slope_lookback"] = config.slope_lookback
    st.session_state["initial_capital"] = config.initial_capital
    st.session_state["fee_rate"] = config.fee_rate
    st.session_state["slippage"] = config.slippage
    st.session_state["fee_rate_percent"] = round(config.fee_rate * 100, 4)
    st.session_state["slippage_percent"] = round(config.slippage * 100, 4)


def _initialize_session_state() -> None:
    if st.session_state.get("ui_initialized"):
        return

    try:
        last_state = preset_manager.load_last_state()
    except PresetError as exc:
        st.warning(f"Unable to restore last session state: {exc}")
        last_state = None

    config = last_state or UIConfig()
    _apply_config_to_session(config)

    st.session_state.setdefault("active_preset", None)
    st.session_state.setdefault("preset_selection", PRESET_PLACEHOLDER)
    st.session_state.setdefault("preset_name_input", "")
    st.session_state.setdefault("preset_error_reported", False)
    st.session_state.setdefault("last_state_error_reported", False)
    st.session_state["ui_initialized"] = True


def _current_config_from_session() -> UIConfig:
    defaults = UIConfig()
    return UIConfig(
        symbol=st.session_state.get("symbol", defaults.symbol),
        interval=st.session_state.get("interval", defaults.interval),
        start_date=st.session_state.get("start_date", defaults.start_date),
        end_date=st.session_state.get("end_date", defaults.end_date),
        force_refresh=st.session_state.get("force_refresh", defaults.force_refresh),
        min_trend_hours=float(st.session_state.get("min_trend_hours", defaults.min_trend_hours)),
        cutoff_scale=float(st.session_state.get("cutoff_scale", defaults.cutoff_scale)),
        stop_type=st.session_state.get("stop_type", defaults.stop_type),
        atr_period=int(st.session_state.get("atr_period", defaults.atr_period)),
        residual_window=int(st.session_state.get("residual_window", defaults.residual_window)),
        k_stop=float(st.session_state.get("k_stop", defaults.k_stop)),
        k_profit=float(st.session_state.get("k_profit", defaults.k_profit)),
        slope_threshold=float(st.session_state.get("slope_threshold", defaults.slope_threshold)),
        slope_lookback=int(st.session_state.get("slope_lookback", defaults.slope_lookback)),
        initial_capital=float(st.session_state.get("initial_capital", defaults.initial_capital)),
        fee_rate=float(st.session_state.get("fee_rate", defaults.fee_rate)),
        slippage=float(st.session_state.get("slippage", defaults.slippage)),
    )


def _live_store() -> dict[str, Any]:
    store = st.session_state.get("_live_state")
    if store is None:
        store = {
            "enabled": False,
            "controller": None,
            "symbol": None,
            "interval": None,
            "start_time": None,
            "config": None,
        }
        st.session_state["_live_state"] = store
    return store


def _shutdown_live_controller() -> None:
    store = _live_store()
    controller = store.get("controller")
    if controller is not None:
        controller.shutdown()
    store.update(
        {
            "enabled": False,
            "controller": None,
            "symbol": None,
            "interval": None,
            "start_time": None,
            "config": None,
        }
    )


def _ensure_live_controller(symbol: str, interval: str, df: pd.DataFrame) -> LiveDataCoordinator:
    if df.empty:
        raise ValueError("Live controller requires non-empty data.")

    store = _live_store()
    start_ts = pd.Timestamp(df["open_time"].min())
    start_dt = start_ts.to_pydatetime()

    controller: LiveDataCoordinator | None = store.get("controller")
    if controller is not None:
        if (
            store.get("symbol") != symbol
            or store.get("interval") != interval
            or store.get("start_time") != start_dt
        ):
            controller.shutdown()
            controller = None

    if controller is None:
        controller = LiveDataCoordinator(
            symbol=symbol,
            interval=interval,
            start_time=start_dt,
            initial_data=df.copy(),
        )
        controller.start()

    store.update(
        {
            "enabled": True,
            "controller": controller,
            "symbol": symbol,
            "interval": interval,
            "start_time": start_dt,
        }
    )

    return controller


def _list_presets() -> list[str]:
    try:
        presets = preset_manager.list_presets()
    except PresetError as exc:
        if not st.session_state.get("preset_error_reported"):
            st.warning(f"Unable to load saved presets: {exc}")
            st.session_state["preset_error_reported"] = True
        return []
    else:
        st.session_state["preset_error_reported"] = False
        return presets


def _persist_last_state(config: UIConfig) -> None:
    try:
        preset_manager.save_last_state(config)
    except PresetError as exc:
        if not st.session_state.get("last_state_error_reported"):
            st.warning(f"Unable to persist last session state: {exc}")
            st.session_state["last_state_error_reported"] = True
    else:
        st.session_state["last_state_error_reported"] = False


st.set_page_config(page_title="Binance Fourier Backtester", layout="wide")

_initialize_session_state()

st.title("Binance Fourier Backtester")

st.markdown("---")

with st.expander("💾 Presets & Persistence", expanded=False):
    available_presets = _list_presets()
    preset_options = [PRESET_PLACEHOLDER, *available_presets]
    current_selection = st.session_state.get("preset_selection", PRESET_PLACEHOLDER)
    preset_selection = st.selectbox(
        "Saved presets",
        preset_options,
        index=_safe_index(preset_options, current_selection),
        key="preset_selection",
    )

    preset_name = st.text_input(
        "Preset name",
        value=st.session_state.get("preset_name_input", ""),
        key="preset_name_input",
        placeholder="e.g. Trend following ATR",
    )

    load_col, save_col, delete_col = st.columns(3)

    with load_col:
        if st.button("Load preset", use_container_width=True):
            if preset_selection == PRESET_PLACEHOLDER:
                st.info("Select a preset to load.")
            else:
                try:
                    config = preset_manager.load_preset(preset_selection)
                except PresetError as exc:
                    st.error(f"Failed to load preset '{preset_selection}': {exc}")
                else:
                    _apply_config_to_session(config)
                    st.session_state["active_preset"] = preset_selection
                    st.session_state["preset_name_input"] = preset_selection
                    st.success(f"Loaded preset '{preset_selection}'")

    with save_col:
        if st.button("Save preset", use_container_width=True):
            try:
                config = _current_config_from_session()
                preset_manager.save_preset(preset_name, config)
            except PresetError as exc:
                st.error(f"Unable to save preset: {exc}")
            else:
                saved_name = preset_name.strip()
                st.success(f"Saved preset '{saved_name}'")
                st.session_state["active_preset"] = saved_name
                st.session_state["preset_selection"] = saved_name
                st.session_state["preset_name_input"] = saved_name

    with delete_col:
        if st.button("Delete preset", use_container_width=True):
            if preset_selection == PRESET_PLACEHOLDER:
                st.info("Select a preset to delete.")
            else:
                try:
                    preset_manager.delete_preset(preset_selection)
                except PresetError as exc:
                    st.error(f"Unable to delete preset: {exc}")
                else:
                    st.success(f"Deleted preset '{preset_selection}'")
                    if st.session_state.get("active_preset") == preset_selection:
                        st.session_state["active_preset"] = None
                    st.session_state["preset_selection"] = PRESET_PLACEHOLDER
                    st.session_state["preset_name_input"] = ""

    active_label = st.session_state.get("active_preset") or "Last session"
    st.caption(f"Active preset: {active_label}")
    st.caption("Latest parameter choices are automatically stored as your last session.")

st.subheader("📊 Data Configuration")

col1, col2, col3, col4 = st.columns(4)

with col1:
    symbol = st.selectbox(
        "Symbol",
        SYMBOL_OPTIONS,
        index=_safe_index(SYMBOL_OPTIONS, st.session_state.get("symbol")),
        key="symbol",
    )

with col2:
    interval = st.selectbox(
        "Interval",
        INTERVAL_OPTIONS,
        index=_safe_index(INTERVAL_OPTIONS, st.session_state.get("interval")),
        key="interval",
    )

with col3:
    start_date = st.date_input(
        "Start Date",
        value=st.session_state.get("start_date"),
        key="start_date",
    )

with col4:
    end_date = st.date_input(
        "End Date",
        value=st.session_state.get("end_date"),
        key="end_date",
    )

force_refresh = st.checkbox(
    "Force Refresh (bypass cache)",
    value=bool(st.session_state.get("force_refresh", False)),
    key="force_refresh",
)

if st.button("Load Data"):
    start_date_value = st.session_state.get("start_date")
    end_date_value = st.session_state.get("end_date")

    if start_date_value is None or end_date_value is None:
        st.error("Both start and end dates must be provided.")
    elif start_date_value > end_date_value:
        st.error("Start date must be earlier than end date.")
    else:
        _shutdown_live_controller()
        st.session_state["live_enabled"] = False
        with st.spinner("Loading data..."):
            start_dt = datetime.combine(start_date_value, datetime.min.time()).replace(tzinfo=UTC)
            end_dt = datetime.combine(end_date_value, datetime.max.time()).replace(tzinfo=UTC)
            try:
                df = load_klines(
                    symbol=symbol,
                    interval=interval,
                    start=start_dt,
                    end=end_dt,
                    force_refresh=force_refresh,
                )
            except ValueError as exc:
                st.error(str(exc))
            except BinanceRateLimitError as exc:
                wait_hint = f" Please retry in ~{int(exc.retry_after)} seconds." if exc.retry_after else ""
                st.error(f"Binance rate limit reached.{wait_hint}")
                if exc.used_weight is not None:
                    st.caption(f"Current 1 minute weight usage: {exc.used_weight}")
                st.info("Tip: Narrow the date range or disable force refresh to reduce API requests.")
            except BinanceRequestError as exc:
                st.error(f"Binance rejected the request: {exc}")
            except BinanceTransientError as exc:
                st.warning(f"Temporary issue fetching data: {exc}. Please try again.")
            except Exception:
                st.error("Unexpected error while loading data.")
                st.code(traceback.format_exc())
            else:
                st.session_state["data"] = df
                st.session_state["backtest_result"] = None
                if df.empty:
                    st.warning("No data returned for the selected range. Try adjusting the dates.")
                else:
                    st.success(f"Loaded {len(df)} candles")

if "data" in st.session_state and not st.session_state["data"].empty:
    df = st.session_state["data"]

    live_store = _live_store()
    st.session_state.setdefault("live_enabled", live_store.get("enabled", False))

    st.markdown("---")
    st.subheader("⚡ Live Data")

    live_cols = st.columns([1, 1, 2])
    with live_cols[0]:
        live_toggle = st.toggle("Live updates", key="live_enabled")
    with live_cols[1]:
        manual_refresh_clicked = st.button("Manual Refresh", key="live_manual_refresh")
    status_col = live_cols[2]
    status_rendered = False

    controller = live_store.get("controller")
    manual_refresh_requested = False

    if not live_toggle and live_store.get("enabled"):
        _shutdown_live_controller()
        controller = None
        live_toggle = False
    elif live_toggle:
        try:
            controller = _ensure_live_controller(symbol, interval, df)
        except ValueError as exc:
            status_col.warning(str(exc))
            status_rendered = True
            st.session_state["live_enabled"] = False
            live_toggle = False
            controller = None

    live_store["enabled"] = live_toggle

    if manual_refresh_clicked:
        if controller is not None:
            controller.manual_refresh()
            manual_refresh_requested = True
        else:
            status_col.info("Enable live updates to refresh data.")
            status_rendered = True

    snapshot = None
    if controller is not None and live_toggle:
        snapshot = controller.snapshot(force_full=manual_refresh_requested)
        live_store["last_snapshot"] = snapshot
        status = snapshot.status

        if status.state == StreamState.CONNECTED:
            status_col.markdown("🟢 **Connected**")
        elif status.state == StreamState.RECONNECTING:
            status_col.markdown("🟠 **Reconnecting**")
        elif status.state == StreamState.CONNECTING:
            status_col.markdown("🟡 **Connecting**")
        else:
            status_col.markdown("⚪️ **Stopped**")

        status_rendered = True

        if status.last_event_time is not None:
            ts = status.last_event_time.astimezone(UTC)
            status_col.caption(f"Last update: {ts:%Y-%m-%d %H:%M:%S %Z}")

        if status.last_error:
            st.warning(f"Live stream issue: {status.last_error}")

        st.session_state["data"] = snapshot.data
        df = snapshot.data

        if snapshot.result is None and not controller.has_engine():
            status_col.caption("Run a backtest to enable live signal recomputation.")
        elif snapshot.result is not None:
            st.session_state["smoothed"] = snapshot.result.smoothed
            st.session_state["long_stop"] = snapshot.result.long_stop
            st.session_state["long_profit"] = snapshot.result.long_profit
            st.session_state["signals"] = snapshot.result.signals

            if live_store.get("config") is not None and snapshot.used_full_recompute:
                config_obj = st.session_state.get("config")
                if config_obj is not None and len(df) > 0:
                    backtest_result = run_backtest(
                        signals=snapshot.result.signals,
                        open_prices=df["open"].values,
                        high_prices=df["high"].values,
                        low_prices=df["low"].values,
                        close_prices=df["close"].values,
                        timestamps=df["open_time"],
                        config=config_obj,
                    )
                    st.session_state["backtest_result"] = backtest_result
    if not status_rendered:
        status_col.info("Live updates disabled.")

    st.markdown("---")
    st.subheader("⚙️ Backtest Configuration")

    interval_hours = {"30m": 0.5, "1h": 1.0, "4h": 4.0}
    hours_per_bar = interval_hours.get(interval, 1.0)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        min_trend_hours = st.number_input(
            "Min Trend Period (hours)",
            min_value=1.0,
            max_value=720.0,
            value=float(st.session_state.get("min_trend_hours", 24.0)),
            step=1.0,
            key="min_trend_hours",
        )
    min_trend_bars = max(1, int(min_trend_hours / hours_per_bar))

    with col2:
        cutoff_scale = st.slider(
            "Cutoff Scale",
            min_value=0.5,
            max_value=3.0,
            value=float(st.session_state.get("cutoff_scale", 1.0)),
            step=0.1,
            help="Higher = more aggressive smoothing",
            key="cutoff_scale",
        )

    with col3:
        stop_type = st.selectbox(
            "Stop Type",
            STOP_TYPES,
            index=_safe_index(STOP_TYPES, st.session_state.get("stop_type")),
            key="stop_type",
        )

    with col4:
        atr_period_default = int(st.session_state.get("atr_period", 14))
        residual_window_default = int(st.session_state.get("residual_window", 20))
        if stop_type == "ATR":
            atr_period = int(
                st.number_input(
                    "ATR Period",
                    min_value=5,
                    max_value=50,
                    value=atr_period_default,
                    step=1,
                    key="atr_period",
                )
            )
            residual_window = residual_window_default
        else:
            residual_window = int(
                st.number_input(
                    "Residual Window",
                    min_value=5,
                    max_value=100,
                    value=residual_window_default,
                    step=5,
                    key="residual_window",
                )
            )
            atr_period = atr_period_default

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        k_stop = st.slider(
            "Stop Loss Multiplier",
            min_value=0.5,
            max_value=5.0,
            value=float(st.session_state.get("k_stop", 2.0)),
            step=0.5,
            key="k_stop",
        )

    with col2:
        k_profit = st.slider(
            "Take Profit Multiplier",
            min_value=1.0,
            max_value=10.0,
            value=float(st.session_state.get("k_profit", 3.0)),
            step=0.5,
            key="k_profit",
        )

    with col3:
        slope_threshold = st.number_input(
            "Slope Threshold",
            min_value=0.0,
            max_value=10.0,
            value=float(st.session_state.get("slope_threshold", 0.0)),
            step=0.1,
            help="Minimum slope for entry",
            key="slope_threshold",
        )

    with col4:
        slope_lookback = int(
            st.number_input(
                "Slope Lookback (bars)",
                min_value=1,
                max_value=10,
                value=int(st.session_state.get("slope_lookback", 1)),
                step=1,
                key="slope_lookback",
            )
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        initial_capital = st.number_input(
            "Initial Capital",
            min_value=1000.0,
            max_value=1_000_000.0,
            value=float(st.session_state.get("initial_capital", 10_000.0)),
            step=1000.0,
            key="initial_capital",
        )

    with col2:
        fee_rate_percent_default = float(
            st.session_state.get(
                "fee_rate_percent",
                st.session_state.get("fee_rate", settings.default_fee_rate) * 100,
            )
        )
        fee_rate_percent = st.number_input(
            "Fee Rate (%)",
            min_value=0.0,
            max_value=1.0,
            value=fee_rate_percent_default,
            step=0.01,
            key="fee_rate_percent",
        )
        fee_rate = fee_rate_percent / 100.0
        st.session_state["fee_rate"] = fee_rate

    with col3:
        slippage_decimal_default = float(
            st.session_state.get("slippage", settings.default_slippage_bps / 10_000)
        )
        slippage_percent_default = float(
            st.session_state.get("slippage_percent", slippage_decimal_default * 100)
        )
        slippage_percent = st.number_input(
            "Slippage (%)",
            min_value=0.0,
            max_value=1.0,
            value=slippage_percent_default,
            step=0.01,
            key="slippage_percent",
        )
        slippage = slippage_percent / 100.0
        st.session_state["slippage"] = slippage

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

                if st.session_state.get("live_enabled"):
                    live_store = _live_store()
                    controller = live_store.get("controller")
                    if controller is not None:
                        live_config = LiveComputationConfig(
                            min_trend_bars=min_trend_bars,
                            cutoff_scale=cutoff_scale,
                            stop_type=stop_type,
                            atr_period=atr_period,
                            residual_window=residual_window,
                            k_stop=k_stop,
                            k_profit=k_profit,
                            slope_threshold=slope_threshold,
                            slope_lookback=slope_lookback,
                        )
                        live_store["config"] = live_config
                        engine_result = controller.configure(live_config, df)
                        st.session_state["smoothed"] = engine_result.smoothed
                        st.session_state["long_stop"] = engine_result.long_stop
                        st.session_state["long_profit"] = engine_result.long_profit
                        st.session_state["signals"] = engine_result.signals

                        result = run_backtest(
                            signals=engine_result.signals,
                            open_prices=open_prices,
                            high_prices=high,
                            low_prices=low,
                            close_prices=close,
                            timestamps=timestamps,
                            config=config,
                        )
                        st.session_state["backtest_result"] = result

                st.success(f"Backtest complete! {result.metrics['n_trades']} trades executed.")

            except Exception as e:
                st.error(f"Error running backtest: {e}")
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

current_config = _current_config_from_session()
_persist_last_state(current_config)

live_state = st.session_state.get("_live_state")
if (
    st.session_state.get("live_enabled")
    and live_state
    and live_state.get("controller") is not None
):
    time.sleep(settings.live_ui_poll_interval_seconds)
    st.experimental_rerun()
