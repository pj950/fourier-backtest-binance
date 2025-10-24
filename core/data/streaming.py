from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import pandas as pd
from websocket import WebSocketApp

from config.settings import settings
from core.data.cache import KlineCache
from core.data.exceptions import BinanceStreamError
from core.utils.time import now_utc, timestamp_to_datetime

_DEFAULT_COLUMNS: list[str] = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "close_time",
]


class StreamState(str, Enum):
    STOPPED = "stopped"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


@dataclass(frozen=True)
class KlineStreamUpdate:
    open_time: datetime
    close_time: datetime
    is_final: bool
    source: str


@dataclass(frozen=True)
class StreamStatus:
    state: StreamState
    last_error: str | None
    reconnect_attempts: int
    last_event_time: datetime | None


def _build_stream_url(symbol: str, interval: str) -> str:
    base_url = settings.binance_ws_base_url.rstrip("/")
    stream = f"{symbol.lower()}@kline_{interval}"
    return f"{base_url}/{stream}"


def _row_from_kline_payload(kline: dict[str, Any]) -> tuple[pd.DataFrame, bool]:
    required_keys = {"t", "T", "o", "h", "l", "c", "v", "q", "n"}
    if not required_keys.issubset(kline):
        missing = ", ".join(sorted(required_keys - set(kline)))
        raise BinanceStreamError(f"Missing fields in kline payload: {missing}")

    try:
        open_time = timestamp_to_datetime(int(kline["t"]))
        close_time = timestamp_to_datetime(int(kline["T"]))
        row = pd.DataFrame(
            {
                "open_time": [pd.Timestamp(open_time, tz=UTC)],
                "open": [float(kline["o"])],
                "high": [float(kline["h"])],
                "low": [float(kline["l"])],
                "close": [float(kline["c"])],
                "volume": [float(kline["v"])],
                "quote_volume": [float(kline["q"])],
                "trades": [int(kline["n"])],
                "close_time": [pd.Timestamp(close_time, tz=UTC)],
            }
        )
    except (TypeError, ValueError) as exc:
        raise BinanceStreamError(f"Invalid kline payload: {kline}") from exc

    is_final = bool(kline.get("x", False))
    return row, is_final


class BinanceKlineStreamer:
    def __init__(
        self,
        symbol: str,
        interval: str,
        *,
        cache: KlineCache | None = None,
        start_time: datetime | None = None,
        initial_data: pd.DataFrame | None = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.interval = interval
        self._cache = cache or KlineCache(symbol=self.symbol, interval=self.interval)
        self._start_time = start_time

        self._lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._updates_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._thread: threading.Thread | None = None
        self._ws: WebSocketApp | None = None

        self._state: StreamState = StreamState.STOPPED
        self._last_error: str | None = None
        self._reconnect_attempts = 0
        self._last_event_time: datetime | None = None
        self._last_backfill = now_utc()

        self._updates: list[KlineStreamUpdate] = []

        self._data = self._prepare_initial_data(initial_data)

    def _prepare_initial_data(self, initial: pd.DataFrame | None) -> pd.DataFrame:
        if initial is None or initial.empty:
            return pd.DataFrame(columns=_DEFAULT_COLUMNS)

        prepared = (
            initial[_DEFAULT_COLUMNS]
            .sort_values("open_time")
            .drop_duplicates(subset=["open_time"], keep="last")
            .reset_index(drop=True)
        )
        return self._filter_range(prepared)

    def _filter_range(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        filtered = df
        if self._start_time is not None:
            filtered = filtered[filtered["open_time"] >= self._start_time]
        return filtered.sort_values("open_time").reset_index(drop=True)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._seed_cache()

        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"{self.symbol}-{self.interval}-stream",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop_event.set()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:  # pragma: no cover - best effort
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        self._set_state(StreamState.STOPPED)

    def _seed_cache(self) -> None:
        snapshot = self.snapshot()
        if not snapshot.empty:
            combined = self._cache.merge_and_save(snapshot)
            with self._lock:
                self._data = self._filter_range(combined)
        else:
            cached = self._cache.load()
            if cached is not None and not cached.empty:
                with self._lock:
                    self._data = self._filter_range(cached)
        self._last_backfill = now_utc()

    def snapshot(self) -> pd.DataFrame:
        with self._lock:
            return self._data.copy()

    def drain_updates(self) -> list[KlineStreamUpdate]:
        with self._updates_lock:
            updates = self._updates
            self._updates = []
        return updates

    def status(self) -> StreamStatus:
        with self._status_lock:
            return StreamStatus(
                state=self._state,
                last_error=self._last_error,
                reconnect_attempts=self._reconnect_attempts,
                last_event_time=self._last_event_time,
            )

    def wait_for_update(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.time() + timeout
        while True:
            with self._updates_lock:
                if self._updates:
                    return True
            if deadline is not None and time.time() >= deadline:
                return False
            time.sleep(0.05)

    def perform_backfill(self) -> None:
        with self._lock:
            if self._data.empty:
                previous_last_open: datetime | None = None
                previous_last_row: pd.Series | None = None
            else:
                previous_last_open = pd.Timestamp(self._data.iloc[-1]["open_time"]).to_pydatetime()
                previous_last_row = self._data.iloc[-1].copy()

        try:
            refreshed = self._cache.incremental_update()
        except Exception as exc:  # pragma: no cover - network failures
            self._set_error(f"Backfill failed: {exc}")
            return

        if refreshed.empty:
            self._last_backfill = now_utc()
            return

        filtered = self._filter_range(refreshed)
        last_row = filtered.iloc[-1]
        last_open = pd.Timestamp(last_row["open_time"]).to_pydatetime()

        with self._lock:
            self._data = filtered

        if previous_last_open == last_open:
            if previous_last_row is not None and last_row.equals(previous_last_row):
                self._last_backfill = now_utc()
                return

        self._register_update(
            KlineStreamUpdate(
                open_time=pd.Timestamp(last_row["open_time"]).to_pydatetime(),
                close_time=pd.Timestamp(last_row["close_time"]).to_pydatetime(),
                is_final=True,
                source="rest",
            )
        )
        self._last_backfill = now_utc()

    def _run_loop(self) -> None:
        url = _build_stream_url(self.symbol, self.interval)
        while not self._stop_event.is_set():
            self._set_state(StreamState.CONNECTING)
            ws = WebSocketApp(
                url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self._ws = ws
            try:
                ws.run_forever(
                    ping_interval=settings.binance_ws_ping_interval,
                    ping_timeout=settings.binance_ws_ping_interval / 2.0,
                )
            except Exception as exc:  # pragma: no cover - best effort logging
                self._set_error(f"WebSocket error: {exc}")
            finally:
                self._ws = None

            if self._stop_event.is_set():
                break

            self._set_state(StreamState.RECONNECTING)
            self._reconnect_attempts += 1
            backoff_seconds = min(
                settings.binance_ws_backoff_initial * (2 ** (self._reconnect_attempts - 1)),
                settings.binance_ws_backoff_max,
            )
            time.sleep(backoff_seconds)

        self._set_state(StreamState.STOPPED)

    def _on_open(self, _ws: WebSocketApp) -> None:
        with self._status_lock:
            self._state = StreamState.CONNECTED
            self._last_error = None
            self._reconnect_attempts = 0
        self._ensure_recent_backfill()

    def _on_close(self, _ws: WebSocketApp, _code: int, _msg: str) -> None:
        with self._status_lock:
            if not self._stop_event.is_set():
                self._state = StreamState.RECONNECTING

    def _on_error(self, _ws: WebSocketApp, error: Exception) -> None:
        self._set_error(str(error))

    def _ensure_recent_backfill(self) -> None:
        if (now_utc() - self._last_backfill) > timedelta(seconds=settings.live_backfill_interval_seconds):
            self.perform_backfill()

    def _on_message(self, _ws: WebSocketApp, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return

        kline = payload.get("k")
        if not isinstance(kline, dict):
            return

        try:
            row, is_final = _row_from_kline_payload(kline)
        except BinanceStreamError as exc:
            self._set_error(str(exc))
            return

        self._apply_row(row, is_final, source="ws")
        self._ensure_recent_backfill()

    def _apply_row(self, row: pd.DataFrame, is_final: bool, *, source: str) -> None:
        open_time = row.iloc[0]["open_time"]
        close_time = row.iloc[0]["close_time"]

        with self._lock:
            if self._data.empty:
                updated = row.copy()
            else:
                mask = self._data["open_time"] == open_time
                if mask.any():
                    for column in row.columns:
                        self._data.loc[mask, column] = row.iloc[0][column]
                    updated = self._data
                else:
                    updated = pd.concat([self._data, row], ignore_index=True, sort=False)
            self._data = self._filter_range(updated)

        self._last_event_time = pd.Timestamp(close_time).to_pydatetime()

        if is_final and source == "ws":
            self._cache.merge_and_save(row)

        self._register_update(
            KlineStreamUpdate(
                open_time=pd.Timestamp(open_time).to_pydatetime(),
                close_time=pd.Timestamp(close_time).to_pydatetime(),
                is_final=is_final,
                source=source,
            )
        )

    def _register_update(self, update: KlineStreamUpdate) -> None:
        with self._updates_lock:
            self._updates.append(update)

    def _set_state(self, state: StreamState) -> None:
        with self._status_lock:
            self._state = state

    def _set_error(self, message: str) -> None:
        with self._status_lock:
            self._last_error = message

    # Exposed for testing purposes
    def _handle_kline_payload(self, kline: dict[str, Any]) -> None:
        row, is_final = _row_from_kline_payload(kline)
        self._apply_row(row, is_final, source="ws")
