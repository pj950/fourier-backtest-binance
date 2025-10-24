from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import yaml


class PresetError(RuntimeError):
    """Raised when preset operations fail."""


@dataclass(slots=True)
class UIConfig:
    """Serializable representation of the Streamlit UI configuration."""

    symbol: str = "BTCUSDT"
    interval: str = "1h"
    start_date: date = field(default_factory=lambda: date(2024, 1, 1))
    end_date: date = field(default_factory=date.today)
    force_refresh: bool = False
    min_trend_hours: float = 24.0
    cutoff_scale: float = 1.0
    stop_type: str = "ATR"
    atr_period: int = 14
    residual_window: int = 20
    k_stop: float = 2.0
    k_profit: float = 3.0
    slope_threshold: float = 0.0
    slope_lookback: int = 1
    initial_capital: float = 10_000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005

    def to_serializable_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start_date"] = self.start_date.isoformat()
        data["end_date"] = self.end_date.isoformat()
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "UIConfig":
        if not data:
            return cls()

        defaults = asdict(cls())
        merged = {**defaults, **dict(data)}

        def _parse_date(value: Any, fallback: date) -> date:
            if value is None:
                return fallback
            if isinstance(value, date) and not isinstance(value, datetime):
                return value
            if isinstance(value, datetime):
                return value.date()
            if isinstance(value, str):
                try:
                    return date.fromisoformat(value)
                except ValueError as exc:  # pragma: no cover - defensive guard
                    raise PresetError(f"Invalid date value: {value}") from exc
            raise PresetError(f"Unsupported date value: {value!r}")

        merged["start_date"] = _parse_date(merged.get("start_date"), defaults["start_date"])
        merged["end_date"] = _parse_date(merged.get("end_date"), defaults["end_date"])

        valid_kwargs = {field: merged[field] for field in cls.__annotations__ if field in merged}
        return cls(**valid_kwargs)


class PresetManager:
    """Handles persistence of UI presets and the last-session state."""

    def __init__(self, presets_path: Path, last_state_path: Path) -> None:
        self.presets_path = presets_path
        self.last_state_path = last_state_path
        self.presets_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_state_path.parent.mkdir(parents=True, exist_ok=True)

    def list_presets(self) -> list[str]:
        presets = self._load_yaml(self.presets_path)
        return sorted(presets.keys())

    def load_preset(self, name: str) -> UIConfig:
        presets = self._load_yaml(self.presets_path)
        raw = presets.get(name)
        if raw is None:
            raise PresetError(f"Preset '{name}' not found")
        if not isinstance(raw, Mapping):
            raise PresetError(f"Preset '{name}' has invalid structure")
        return UIConfig.from_mapping(raw)

    def save_preset(self, name: str, config: UIConfig) -> None:
        safe_name = name.strip()
        if not safe_name:
            raise PresetError("Preset name cannot be empty")
        if len(safe_name) > 64:
            raise PresetError("Preset name must be 64 characters or fewer")
        if any(char in "/\\" for char in safe_name):
            raise PresetError("Preset name cannot contain path separators")

        presets = self._load_yaml(self.presets_path)
        presets[safe_name] = config.to_serializable_dict()
        self._write_yaml(self.presets_path, presets)

    def delete_preset(self, name: str) -> None:
        presets = self._load_yaml(self.presets_path)
        if name in presets:
            presets.pop(name)
            self._write_yaml(self.presets_path, presets)

    def load_last_state(self) -> UIConfig | None:
        data = self._load_yaml(self.last_state_path)
        if not data:
            return None
        if not isinstance(data, Mapping):
            raise PresetError("Last session state file is corrupted")
        return UIConfig.from_mapping(data)

    def save_last_state(self, config: UIConfig) -> None:
        self._write_yaml(self.last_state_path, config.to_serializable_dict())

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:
            raise PresetError(f"Failed to parse YAML file: {path}") from exc
        except OSError as exc:
            raise PresetError(f"Failed to read {path}: {exc}") from exc

        if not isinstance(loaded, Mapping):
            raise PresetError(f"Invalid data format in {path}")

        return dict(loaded)

    @staticmethod
    def _write_yaml(path: Path, data: Mapping[str, Any]) -> None:
        try:
            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(dict(data), handle, sort_keys=True, allow_unicode=True)
        except yaml.YAMLError as exc:  # pragma: no cover - unlikely
            raise PresetError(f"Failed to serialize presets to {path}") from exc
        except OSError as exc:  # pragma: no cover - unlikely
            raise PresetError(f"Failed to write {path}: {exc}") from exc
