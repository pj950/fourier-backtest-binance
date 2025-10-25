import numpy as np
import pandas as pd
import pytest

from core.portfolio.weights import (
    apply_weight_caps,
    compute_equal_weights,
    compute_market_cap_weights,
    compute_risk_parity_weights,
    compute_volatility_scaled_weights,
    rebalance_weights,
)


def test_compute_equal_weights():
    """Test equal weight calculation."""
    weights = compute_equal_weights(4)
    assert len(weights) == 4
    assert np.allclose(weights, 0.25)
    assert np.isclose(weights.sum(), 1.0)


def test_compute_equal_weights_empty():
    """Test equal weights with zero assets."""
    weights = compute_equal_weights(0)
    assert len(weights) == 0


def test_compute_volatility_scaled_weights():
    """Test volatility-scaled weights."""
    # Create synthetic returns with different volatilities
    np.random.seed(42)
    returns = pd.DataFrame({
        "A": np.random.normal(0, 0.01, 100),  # Low vol
        "B": np.random.normal(0, 0.03, 100),  # High vol
    })

    weights = compute_volatility_scaled_weights(returns, lookback=50)

    # Lower vol asset should have higher weight
    assert weights[0] > weights[1]
    assert np.isclose(weights.sum(), 1.0)
    assert len(weights) == 2


def test_compute_volatility_scaled_weights_with_caps():
    """Test volatility weights with caps."""
    np.random.seed(42)
    returns = pd.DataFrame({
        "A": np.random.normal(0, 0.01, 100),
        "B": np.random.normal(0, 0.03, 100),
    })

    weights = compute_volatility_scaled_weights(
        returns,
        lookback=50,
        min_weight=0.3,
        max_weight=0.7,
    )

    assert np.all(weights >= 0.3)
    assert np.all(weights <= 0.7)
    assert np.isclose(weights.sum(), 1.0)


def test_compute_risk_parity_weights():
    """Test risk parity weights."""
    np.random.seed(42)
    returns = pd.DataFrame({
        "A": np.random.normal(0, 0.01, 100),
        "B": np.random.normal(0, 0.02, 100),
        "C": np.random.normal(0, 0.03, 100),
    })

    weights = compute_risk_parity_weights(returns, lookback=60)

    assert len(weights) == 3
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0)

    # Higher vol assets should have lower weights
    assert weights[0] > weights[2]


def test_compute_risk_parity_weights_convergence():
    """Test risk parity converges."""
    np.random.seed(42)
    returns = pd.DataFrame({
        "A": np.random.normal(0, 0.02, 100),
        "B": np.random.normal(0, 0.02, 100),
    })

    weights = compute_risk_parity_weights(
        returns,
        lookback=60,
        max_iterations=50,
        tolerance=1e-6,
    )

    # Similar volatility should lead to similar weights
    assert np.isclose(weights[0], weights[1], atol=0.1)


def test_apply_weight_caps():
    """Test weight capping and normalization."""
    weights = np.array([0.1, 0.6, 0.3])

    capped = apply_weight_caps(weights, min_weight=0.2, max_weight=0.5)

    assert np.all(capped >= 0.2)
    assert np.all(capped <= 0.5)
    assert np.isclose(capped.sum(), 1.0)


def test_apply_weight_caps_empty():
    """Test weight capping with empty array."""
    weights = np.array([])
    capped = apply_weight_caps(weights, min_weight=0.1, max_weight=0.9)
    assert len(capped) == 0


def test_compute_market_cap_weights():
    """Test market cap weighted portfolio."""
    market_caps = {
        "BTC": 1000.0,
        "ETH": 500.0,
        "SOL": 100.0,
    }
    symbols = ["BTC", "ETH", "SOL"]

    weights = compute_market_cap_weights(market_caps, symbols)

    # BTC should have highest weight
    assert weights[0] > weights[1] > weights[2]
    assert np.isclose(weights.sum(), 1.0)


def test_compute_market_cap_weights_with_caps():
    """Test market cap weights with constraints."""
    market_caps = {
        "BTC": 1000.0,
        "ETH": 100.0,
    }
    symbols = ["BTC", "ETH"]

    weights = compute_market_cap_weights(
        market_caps,
        symbols,
        max_weight=0.6,
    )

    assert np.all(weights <= 0.6)
    assert np.isclose(weights.sum(), 1.0)


def test_rebalance_weights_no_change():
    """Test rebalancing when weights are close."""
    current = np.array([0.5, 0.5])
    target = np.array([0.52, 0.48])

    new_weights, should_rebalance = rebalance_weights(
        current,
        target,
        threshold=0.05,
    )

    assert not should_rebalance
    assert np.allclose(new_weights, current)


def test_rebalance_weights_needed():
    """Test rebalancing when weights deviate."""
    current = np.array([0.5, 0.5])
    target = np.array([0.7, 0.3])

    new_weights, should_rebalance = rebalance_weights(
        current,
        target,
        threshold=0.05,
    )

    assert should_rebalance
    assert np.allclose(new_weights, target)


def test_rebalance_weights_size_mismatch():
    """Test rebalancing with different sizes."""
    current = np.array([0.5, 0.5])
    target = np.array([0.33, 0.33, 0.34])

    new_weights, should_rebalance = rebalance_weights(current, target)

    assert should_rebalance
    assert np.allclose(new_weights, target)


def test_volatility_weights_handle_zero_vol():
    """Test volatility weights handle zero volatility."""
    returns = pd.DataFrame({
        "A": np.zeros(100),  # Zero volatility
        "B": np.random.normal(0, 0.02, 100),
    })

    weights = compute_volatility_scaled_weights(returns, lookback=50)

    # Should still produce valid weights
    assert len(weights) == 2
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0)
