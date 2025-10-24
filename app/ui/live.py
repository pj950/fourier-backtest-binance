from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from core.analysis.incremental import IncrementalResult, IncrementalSignalEngine
from core.data.cache import KlineCache
from core.data.streaming import (
    BinanceKlineStreamer,
    KlineStreamUpdate,
    StreamStatus,
)



@dataclass(frozen=True)
class LiveComputationConfig:
    min_trend_bars: int
    cutoff_scale: float
    stop_type: str
    atr_period: int
    residual_window: int
    k_stop: float
    k_profit: float
    slope_threshold: float
    slope_lookback: int


@dataclass(frozen=True)
class LiveSnapshot:
    data: "pd.DataFrame"
    status: StreamStatus
    updates: list[KlineStreamUpdate]
    result: IncrementalResult | None
    used_full_recompute: bool


class LiveDataCoordinator:
    def __init__(
        self,
        symbol: str,
        interval: str,
        *,
        start_time: datetime | None,
        initial_data: "pd.DataFrame",
    ) -> None:
        cache = KlineCache(symbol=symbol, interval=interval)
        dt_start = start_time
        if dt_start is not None and hasattr(dt_start, "tzinfo") and getattr(dt_start, "to_pydatetime", None):
            dt_start = dt_start.to_pydatetime()  # type: ignore[assignment]
        self._streamer = BinanceKlineStreamer(
            symbol=symbol,
            interval=interval,
            cache=cache,
            start_time=dt_start,
            initial_data=initial_data,
        )
        self._engine: IncrementalSignalEngine | None = None
        self._config: LiveComputationConfig | None = None
        self._latest_result: IncrementalResult | None = None

    def start(self) -> None:
        self._streamer.start()

    def stop(self) -> None:
        self._streamer.stop(timeout=1.0)

    def status(self) -> StreamStatus:
        return self._streamer.status()

    def has_engine(self) -> bool:
        return self._engine is not None

    def configure(self, config: LiveComputationConfig, df: "pd.DataFrame") -> IncrementalResult:
        self._config = config
        self._engine = IncrementalSignalEngine(
            min_trend_bars=config.min_trend_bars,
            cutoff_scale=config.cutoff_scale,
            stop_type=config.stop_type,
            atr_period=config.atr_period,
            residual_window=config.residual_window,
            k_stop=config.k_stop,
            k_profit=config.k_profit,
            slope_threshold=config.slope_threshold,
            slope_lookback=config.slope_lookback,
        )
        self._latest_result = self._engine.bootstrap(df)
        return self._latest_result

    def manual_refresh(self) -> None:
        self._streamer.perform_backfill()

    def snapshot(self, *, force_full: bool = False) -> LiveSnapshot:
        df = self._streamer.snapshot()
        updates = self._streamer.drain_updates()
        status = self._streamer.status()

        used_full = False

        if self._engine is not None and self._config is not None:
            should_update = force_full or bool(updates) or self._latest_result is None
            if should_update:
                needs_full = force_full or any(
                    update.is_final or update.source == "rest" for update in updates
                )
                self._latest_result = self._engine.sync(
                    df,
                    force_full=needs_full,
                    is_final_update=any(update.is_final for update in updates),
                )
                used_full = needs_full

        return LiveSnapshot(
            data=df,
            status=status,
            updates=updates,
            result=self._latest_result,
            used_full_recompute=used_full,
        )

    def shutdown(self) -> None:
        self.stop()
        self._engine = None
        self._config = None
        self._latest_result = None
