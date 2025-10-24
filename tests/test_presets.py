from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from config.presets import PresetError, PresetManager, UIConfig


def _manager(tmp_path: Path) -> PresetManager:
    presets_path = tmp_path / "presets.yaml"
    last_state_path = tmp_path / "last_state.yaml"
    return PresetManager(presets_path, last_state_path)


def test_ui_config_serialization_roundtrip() -> None:
    config = UIConfig(
        symbol="ETHUSDT",
        interval="4h",
        start_date=date(2024, 2, 1),
        end_date=date(2024, 2, 28),
        force_refresh=True,
        min_trend_hours=48.0,
        cutoff_scale=1.5,
        stop_type="Residual",
        atr_period=20,
        residual_window=40,
        k_stop=2.5,
        k_profit=4.0,
        slope_threshold=0.4,
        slope_lookback=3,
        initial_capital=50_000.0,
        fee_rate=0.0008,
        slippage=0.0003,
    )

    serialized = config.to_serializable_dict()
    restored = UIConfig.from_mapping(serialized)

    assert restored == config


def test_preset_manager_save_and_load(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    config = UIConfig(symbol="BTCUSDT", interval="1h", k_stop=3.0)

    manager.save_preset("swing", config)
    presets = manager.list_presets()
    assert presets == ["swing"]

    restored = manager.load_preset("swing")
    assert restored.symbol == "BTCUSDT"
    assert restored.k_stop == pytest.approx(3.0)


def test_preset_manager_rejects_invalid_names(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    config = UIConfig()

    with pytest.raises(PresetError):
        manager.save_preset("", config)

    with pytest.raises(PresetError):
        manager.save_preset("../../etc/passwd", config)


def test_last_state_persistence(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    config = UIConfig(symbol="ETHUSDT", interval="30m", min_trend_hours=12.0)

    manager.save_last_state(config)
    restored = manager.load_last_state()

    assert restored is not None
    assert restored.symbol == "ETHUSDT"
    assert restored.min_trend_hours == pytest.approx(12.0)
