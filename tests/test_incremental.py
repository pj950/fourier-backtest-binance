from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd

from core.analysis.fourier import smooth_price_series
from core.analysis.incremental import IncrementalSignalEngine
from core.analysis.signals import generate_signals_with_stops
from core.analysis.stops import compute_atr_stops


def _build_frame(length: int = 48) -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=length, freq="1H", tz=UTC)
    base = np.linspace(100.0, 130.0, num=length)
    frame = pd.DataFrame(
        {
            "open_time": times,
            "open": base,
            "high": base + 1.5,
            "low": base - 1.5,
            "close": base,
            "volume": np.full(length, 10.0),
            "quote_volume": np.full(length, 1000.0),
            "trades": np.arange(length, dtype=int),
            "close_time": times + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        }
    )
    return frame


def test_incremental_engine_partial_update_matches_full() -> None:
    df = _build_frame()

    engine = IncrementalSignalEngine(
        min_trend_bars=12,
        cutoff_scale=1.0,
        stop_type="ATR",
        atr_period=14,
        residual_window=20,
        k_stop=2.0,
        k_profit=3.0,
        slope_threshold=0.1,
        slope_lookback=2,
    )
    engine.bootstrap(df)

    df_partial = df.copy()
    df_partial.loc[df_partial.index[-1], ["close", "high", "low"]] += 5.0

    incremental = engine.sync(df_partial, force_full=False, is_final_update=False)

    full_smoothed = smooth_price_series(
        df_partial["close"].values,
        min_period_bars=12,
        cutoff_scale=1.0,
    )
    assert np.allclose(incremental.smoothed, full_smoothed, atol=1e-3)

    full_long_stop, _, _, _ = compute_atr_stops(
        close=df_partial["close"].values,
        high=df_partial["high"].values,
        low=df_partial["low"].values,
        atr_period=14,
        k_stop=2.0,
        k_profit=3.0,
    )
    assert np.allclose(incremental.long_stop, full_long_stop, atol=1e-3)

    full_signals = generate_signals_with_stops(
        close=df_partial["close"].values,
        smoothed=full_smoothed,
        stop_levels=full_long_stop,
        slope_threshold=0.1,
        slope_lookback=2,
    )
    assert np.array_equal(incremental.signals, full_signals)


def test_incremental_engine_final_update_matches_full() -> None:
    df = _build_frame()
    engine = IncrementalSignalEngine(
        min_trend_bars=12,
        cutoff_scale=1.0,
        stop_type="ATR",
        atr_period=14,
        residual_window=20,
        k_stop=2.0,
        k_profit=3.0,
        slope_threshold=0.1,
        slope_lookback=2,
    )
    engine.bootstrap(df)

    new_row = df.iloc[[-1]].copy()
    new_row["open_time"] = new_row["open_time"] + pd.Timedelta(hours=1)
    new_row["close_time"] = new_row["close_time"] + pd.Timedelta(hours=1)
    new_row[["open", "high", "low", "close"]] += 2.0
    df_extended = pd.concat([df, new_row], ignore_index=True)

    incremental = engine.sync(df_extended, is_final_update=True)

    full_smoothed = smooth_price_series(
        df_extended["close"].values,
        min_period_bars=12,
        cutoff_scale=1.0,
    )
    assert np.allclose(incremental.smoothed, full_smoothed)

    full_long_stop, _, _, _ = compute_atr_stops(
        close=df_extended["close"].values,
        high=df_extended["high"].values,
        low=df_extended["low"].values,
        atr_period=14,
        k_stop=2.0,
        k_profit=3.0,
    )
    assert np.allclose(incremental.long_stop, full_long_stop)

    full_signals = generate_signals_with_stops(
        close=df_extended["close"].values,
        smoothed=full_smoothed,
        stop_levels=full_long_stop,
        slope_threshold=0.1,
        slope_lookback=2,
    )
    assert np.array_equal(incremental.signals, full_signals)
