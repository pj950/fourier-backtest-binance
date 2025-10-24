import numpy as np
import pytest

from core.analysis.sizing import (
    check_pyramid_conditions,
    compute_fixed_risk_size,
    compute_pyramid_size,
    compute_volatility_based_size,
)


def test_compute_volatility_based_size_atr():
    size = compute_volatility_based_size(
        capital=10000.0,
        entry_price=100.0,
        stop_price=95.0,
        volatility=2.0,
        risk_target=0.02,
        max_risk_per_trade=0.05,
        vol_target_mode="atr",
    )

    assert size > 0
    assert size <= 10000.0 / 100.0


def test_compute_volatility_based_size_sigma():
    size = compute_volatility_based_size(
        capital=10000.0,
        entry_price=100.0,
        stop_price=95.0,
        volatility=3.0,
        risk_target=0.02,
        max_risk_per_trade=0.05,
        vol_target_mode="sigma",
    )

    assert size > 0
    assert size <= 10000.0 / 100.0


def test_compute_volatility_based_size_respects_max_risk():
    size = compute_volatility_based_size(
        capital=10000.0,
        entry_price=100.0,
        stop_price=95.0,
        volatility=1.0,
        risk_target=0.10,
        max_risk_per_trade=0.02,
        vol_target_mode="atr",
    )

    max_size = (10000.0 * 0.02) / 5.0
    assert size <= max_size + 1e-6


def test_compute_volatility_based_size_zero_volatility():
    size = compute_volatility_based_size(
        capital=10000.0,
        entry_price=100.0,
        stop_price=95.0,
        volatility=0.0,
        risk_target=0.02,
        max_risk_per_trade=0.05,
        vol_target_mode="atr",
    )

    assert size == 0.0


def test_compute_volatility_based_size_zero_capital():
    size = compute_volatility_based_size(
        capital=0.0,
        entry_price=100.0,
        stop_price=95.0,
        volatility=2.0,
        risk_target=0.02,
        max_risk_per_trade=0.05,
    )

    assert size == 0.0


def test_compute_fixed_risk_size_normal():
    size = compute_fixed_risk_size(
        capital=10000.0,
        entry_price=100.0,
        stop_price=95.0,
        risk_fraction=0.01,
    )

    expected_size = (10000.0 * 0.01) / 5.0
    assert abs(size - expected_size) < 1e-6


def test_compute_fixed_risk_size_tight_stop():
    size = compute_fixed_risk_size(
        capital=10000.0,
        entry_price=100.0,
        stop_price=99.0,
        risk_fraction=0.01,
    )

    expected_size = (10000.0 * 0.01) / 1.0
    assert abs(size - expected_size) < 1e-6


def test_compute_fixed_risk_size_no_stop():
    size = compute_fixed_risk_size(
        capital=10000.0,
        entry_price=100.0,
        stop_price=100.0,
        risk_fraction=0.01,
    )

    assert size == 0.0


def test_compute_fixed_risk_size_zero_capital():
    size = compute_fixed_risk_size(
        capital=0.0,
        entry_price=100.0,
        stop_price=95.0,
        risk_fraction=0.01,
    )

    assert size == 0.0


def test_compute_fixed_risk_size_respects_capital_limit():
    size = compute_fixed_risk_size(
        capital=1000.0,
        entry_price=100.0,
        stop_price=50.0,
        risk_fraction=0.50,
    )

    max_size = 1000.0 / 100.0
    assert size <= max_size


def test_compute_pyramid_size_first_add():
    size = compute_pyramid_size(
        initial_size=10.0,
        current_position=10.0,
        max_pyramids=3,
        pyramid_scale=0.5,
    )

    assert size == 5.0


def test_compute_pyramid_size_second_add():
    size = compute_pyramid_size(
        initial_size=10.0,
        current_position=15.0,
        max_pyramids=3,
        pyramid_scale=0.5,
    )

    assert size == 5.0


def test_compute_pyramid_size_max_reached():
    size = compute_pyramid_size(
        initial_size=10.0,
        current_position=25.0,
        max_pyramids=3,
        pyramid_scale=0.5,
    )

    assert size == 0.0


def test_compute_pyramid_size_no_position():
    size = compute_pyramid_size(
        initial_size=10.0,
        current_position=0.0,
        max_pyramids=3,
        pyramid_scale=0.5,
    )

    assert size == 0.0


def test_check_pyramid_conditions_long_met():
    result = check_pyramid_conditions(
        entry_price=100.0,
        current_price=105.0,
        profit_threshold=0.03,
        direction=1,
    )

    assert result == True


def test_check_pyramid_conditions_long_not_met():
    result = check_pyramid_conditions(
        entry_price=100.0,
        current_price=101.0,
        profit_threshold=0.03,
        direction=1,
    )

    assert result == False


def test_check_pyramid_conditions_short_met():
    result = check_pyramid_conditions(
        entry_price=100.0,
        current_price=95.0,
        profit_threshold=0.03,
        direction=-1,
    )

    assert result == True


def test_check_pyramid_conditions_short_not_met():
    result = check_pyramid_conditions(
        entry_price=100.0,
        current_price=99.0,
        profit_threshold=0.03,
        direction=-1,
    )

    assert result == False


def test_check_pyramid_conditions_zero_entry_price():
    result = check_pyramid_conditions(
        entry_price=0.0,
        current_price=100.0,
        profit_threshold=0.03,
        direction=1,
    )

    assert result == False
