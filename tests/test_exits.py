import numpy as np
import pytest

from core.analysis.exits import (
    check_partial_tp_hit,
    check_time_based_exit,
    combine_exit_conditions,
    compute_partial_tp_levels,
    compute_slope_reversal,
)


def test_check_time_based_exit_not_triggered():
    assert check_time_based_exit(entry_idx=10, current_idx=15, max_bars_held=10) == False


def test_check_time_based_exit_triggered():
    assert check_time_based_exit(entry_idx=10, current_idx=20, max_bars_held=10) == True


def test_check_time_based_exit_exact_limit():
    assert check_time_based_exit(entry_idx=10, current_idx=20, max_bars_held=10) == True


def test_check_time_based_exit_no_position():
    assert check_time_based_exit(entry_idx=-1, current_idx=20, max_bars_held=10) == False


def test_compute_partial_tp_levels_long():
    scales = [(0.02, 0.5), (0.05, 0.3), (0.10, 0.2)]
    levels = compute_partial_tp_levels(entry_price=100.0, direction=1, scales=scales)

    assert len(levels) == 3
    assert levels[0] == (102.0, 0.5)
    assert levels[1] == (105.0, 0.3)
    assert levels[2] == (110.0, 0.2)


def test_compute_partial_tp_levels_short():
    scales = [(0.02, 0.5), (0.05, 0.3)]
    levels = compute_partial_tp_levels(entry_price=100.0, direction=-1, scales=scales)

    assert len(levels) == 2
    assert levels[0] == (98.0, 0.5)
    assert levels[1] == (95.0, 0.3)


def test_compute_partial_tp_levels_empty():
    scales = []
    levels = compute_partial_tp_levels(entry_price=100.0, direction=1, scales=scales)

    assert len(levels) == 0


def test_check_partial_tp_hit_long_first_level():
    tp_levels = [(102.0, 0.5), (105.0, 0.3), (110.0, 0.2)]
    hit_levels = set()

    newly_hit = check_partial_tp_hit(
        current_price=103.0,
        high_price=103.5,
        low_price=101.0,
        tp_levels=tp_levels,
        hit_levels=hit_levels,
        direction=1,
    )

    assert 0 in newly_hit
    assert 1 not in newly_hit
    assert 2 not in newly_hit


def test_check_partial_tp_hit_long_multiple_levels():
    tp_levels = [(102.0, 0.5), (105.0, 0.3), (110.0, 0.2)]
    hit_levels = set()

    newly_hit = check_partial_tp_hit(
        current_price=106.0,
        high_price=107.0,
        low_price=101.0,
        tp_levels=tp_levels,
        hit_levels=hit_levels,
        direction=1,
    )

    assert 0 in newly_hit
    assert 1 in newly_hit
    assert 2 not in newly_hit


def test_check_partial_tp_hit_already_hit():
    tp_levels = [(102.0, 0.5), (105.0, 0.3)]
    hit_levels = {0}

    newly_hit = check_partial_tp_hit(
        current_price=106.0,
        high_price=107.0,
        low_price=101.0,
        tp_levels=tp_levels,
        hit_levels=hit_levels,
        direction=1,
    )

    assert 0 not in newly_hit
    assert 1 in newly_hit


def test_check_partial_tp_hit_short():
    tp_levels = [(98.0, 0.5), (95.0, 0.3)]
    hit_levels = set()

    newly_hit = check_partial_tp_hit(
        current_price=97.0,
        high_price=99.0,
        low_price=96.5,
        tp_levels=tp_levels,
        hit_levels=hit_levels,
        direction=-1,
    )

    assert 0 in newly_hit
    assert 1 not in newly_hit


def test_compute_slope_reversal_positive_to_negative():
    smoothed = np.array([100, 102, 104, 106, 105, 103, 101])

    reversal = compute_slope_reversal(smoothed, lookback=1, threshold=0.5)

    assert reversal[4] == True or reversal[5] == True


def test_compute_slope_reversal_negative_to_positive():
    smoothed = np.array([100, 98, 96, 94, 96, 98, 100])

    reversal = compute_slope_reversal(smoothed, lookback=1, threshold=0.5)

    assert reversal[4] == True or reversal[5] == True


def test_compute_slope_reversal_no_reversal():
    smoothed = np.array([100, 102, 104, 106, 108, 110])

    reversal = compute_slope_reversal(smoothed, lookback=1, threshold=0.5)

    assert np.all(reversal == False)


def test_compute_slope_reversal_small_array():
    smoothed = np.array([100, 102])

    reversal = compute_slope_reversal(smoothed, lookback=2, threshold=0.0)

    assert len(reversal) == 2
    assert np.all(reversal == False)


def test_combine_exit_conditions_single():
    condition1 = np.array([True, False, True, False])

    result = combine_exit_conditions(condition1)

    assert np.array_equal(result, condition1)


def test_combine_exit_conditions_multiple():
    condition1 = np.array([True, False, False, False])
    condition2 = np.array([False, True, False, False])
    condition3 = np.array([False, False, True, False])

    result = combine_exit_conditions(condition1, condition2, condition3)

    expected = np.array([True, True, True, False])
    assert np.array_equal(result, expected)


def test_combine_exit_conditions_all_false():
    condition1 = np.array([False, False, False])
    condition2 = np.array([False, False, False])

    result = combine_exit_conditions(condition1, condition2)

    assert np.all(result == False)


def test_combine_exit_conditions_empty():
    result = combine_exit_conditions()

    assert len(result) == 0
