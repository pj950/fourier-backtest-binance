from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from core.analysis.fourier import smooth_price_series
from core.analysis.signals import generate_signals_with_stops
from core.analysis.stops import compute_atr_stops, compute_residual_stops

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class IncrementalResult:
    smoothed: np.ndarray
    long_stop: np.ndarray
    long_profit: np.ndarray
    signals: np.ndarray


class IncrementalSignalEngine:
    def __init__(
        self,
        *,
        min_trend_bars: int,
        cutoff_scale: float,
        stop_type: str,
        atr_period: int,
        residual_window: int,
        k_stop: float,
        k_profit: float,
        slope_threshold: float,
        slope_lookback: int,
        tail_window: int | None = None,
    ) -> None:
        self._min_trend_bars = max(1, min_trend_bars)
        self._cutoff_scale = cutoff_scale
        normalized_stop_type = stop_type.strip().upper()
        if normalized_stop_type not in {"ATR", "RESIDUAL"}:
            raise ValueError("stop_type must be 'ATR' or 'Residual'")
        self._stop_type = normalized_stop_type
        self._atr_period = max(1, atr_period)
        self._residual_window = max(1, residual_window)
        self._k_stop = k_stop
        self._k_profit = k_profit
        self._slope_threshold = slope_threshold
        self._slope_lookback = max(1, slope_lookback)

        base_context = max(
            int(self._min_trend_bars * 2),
            int(self._atr_period * 3),
            int(self._residual_window * 3),
            int((self._slope_lookback + 1) * 4),
            256,
        )
        candidate_window = tail_window if tail_window is not None else base_context
        self._context_window = max(64, int(candidate_window))

        self._result: IncrementalResult | None = None
        self._length = 0

    def bootstrap(self, df: "pd.DataFrame") -> IncrementalResult:
        self._result = self._compute_full(df)
        return self._result

    def sync(
        self,
        df: "pd.DataFrame",
        *,
        force_full: bool = False,
        is_final_update: bool = False,
    ) -> IncrementalResult:
        length = len(df)
        if (
            force_full
            or self._result is None
            or length != self._length
            or is_final_update
            or length == 0
        ):
            self._result = self._compute_full(df)
            return self._result

        self._result = self._apply_tail(df)
        return self._result

    def _compute_full(self, df: "pd.DataFrame") -> IncrementalResult:
        if df.empty:
            empty_float = np.array([], dtype=float)
            empty_int = np.array([], dtype=int)
            self._length = 0
            return IncrementalResult(empty_float, empty_float, empty_float, empty_int)

        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)

        smoothed = smooth_price_series(
            close,
            min_period_bars=self._min_trend_bars,
            cutoff_scale=self._cutoff_scale,
        )

        if self._stop_type == "ATR":
            long_stop, long_profit, _, _ = compute_atr_stops(
                close=close,
                high=high,
                low=low,
                atr_period=self._atr_period,
                k_stop=self._k_stop,
                k_profit=self._k_profit,
            )
        else:
            long_stop, long_profit, _, _ = compute_residual_stops(
                close=close,
                smoothed=smoothed,
                method="sigma",
                window=self._residual_window,
                k_stop=self._k_stop,
                k_profit=self._k_profit,
            )

        signals = generate_signals_with_stops(
            close=close,
            smoothed=smoothed,
            stop_levels=long_stop,
            slope_threshold=self._slope_threshold,
            slope_lookback=self._slope_lookback,
        )

        self._length = len(close)
        return IncrementalResult(smoothed, long_stop, long_profit, signals)

    def _apply_tail(self, df: "pd.DataFrame") -> IncrementalResult:
        assert self._result is not None

        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)

        length = len(close)
        start = max(0, length - self._context_window)

        close_tail = close[start:]
        high_tail = high[start:]
        low_tail = low[start:]

        effective_min_period = max(1, min(self._min_trend_bars, len(close_tail)))

        smoothed_tail = smooth_price_series(
            close_tail,
            min_period_bars=effective_min_period,
            cutoff_scale=self._cutoff_scale,
        )

        if self._stop_type == "ATR":
            long_stop_tail, long_profit_tail, _, _ = compute_atr_stops(
                close=close_tail,
                high=high_tail,
                low=low_tail,
                atr_period=self._atr_period,
                k_stop=self._k_stop,
                k_profit=self._k_profit,
            )
        else:
            window = max(1, min(self._residual_window, len(close_tail)))
            long_stop_tail, long_profit_tail, _, _ = compute_residual_stops(
                close=close_tail,
                smoothed=smoothed_tail,
                method="sigma",
                window=window,
                k_stop=self._k_stop,
                k_profit=self._k_profit,
            )

        signals_tail = generate_signals_with_stops(
            close=close_tail,
            smoothed=smoothed_tail,
            stop_levels=long_stop_tail,
            slope_threshold=self._slope_threshold,
            slope_lookback=self._slope_lookback,
        )

        smoothed = self._result.smoothed.copy()
        long_stop = self._result.long_stop.copy()
        long_profit = self._result.long_profit.copy()
        signals = self._result.signals.copy()

        smoothed[start:] = smoothed_tail
        long_stop[start:] = long_stop_tail
        long_profit[start:] = long_profit_tail
        signals[start:] = signals_tail

        self._length = len(df)
        return IncrementalResult(smoothed, long_stop, long_profit, signals)
