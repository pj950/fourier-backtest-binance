"""Tests for parameter optimization module."""

import numpy as np
import pandas as pd
import pytest

from core.optimization.monte_carlo import (
    block_bootstrap_returns,
    compute_mc_metrics,
    monte_carlo_equity_curves,
)
from core.optimization.params import ParamSpace, StrategyParams, create_default_param_space
from core.optimization.runner import OptimizationResult, OptimizationRunner
from core.optimization.search import BayesianSearch, GridSearch, RandomSearch
from core.optimization.walkforward import (
    WalkForwardWindow,
    compute_combined_oos_metrics,
    create_walkforward_windows,
    split_data_by_window,
)


def test_param_space_int():
    """Test integer parameter space."""
    space = ParamSpace(name="period", param_type="int", min_val=10, max_val=50)
    assert space.name == "period"
    assert space.param_type == "int"
    assert space.min_val == 10
    assert space.max_val == 50


def test_param_space_float():
    """Test float parameter space."""
    space = ParamSpace(name="threshold", param_type="float", min_val=0.0, max_val=1.0)
    assert space.name == "threshold"
    assert space.param_type == "float"


def test_param_space_categorical():
    """Test categorical parameter space."""
    space = ParamSpace(name="method", param_type="categorical", categories=["A", "B", "C"])
    assert space.name == "method"
    assert space.categories == ["A", "B", "C"]


def test_param_space_validation():
    """Test parameter space validation."""
    with pytest.raises(ValueError):
        ParamSpace(name="bad", param_type="int", min_val=None, max_val=None)

    with pytest.raises(ValueError):
        ParamSpace(name="bad", param_type="int", min_val=10, max_val=5)

    with pytest.raises(ValueError):
        ParamSpace(name="bad", param_type="categorical", categories=None)


def test_strategy_params():
    """Test strategy parameters."""
    params = StrategyParams(
        min_trend_period_hours=48.0,
        cutoff_scale=1.0,
        atr_period=14,
        k_stop=2.0,
        k_profit=3.0,
    )

    assert params.min_trend_period_hours == 48.0
    assert params.atr_period == 14

    params_dict = params.to_dict()
    assert "min_trend_period_hours" in params_dict
    assert "atr_period" in params_dict

    params2 = StrategyParams.from_dict(params_dict)
    assert params2.min_trend_period_hours == params.min_trend_period_hours


def test_create_default_param_space():
    """Test default parameter space creation."""
    spaces = create_default_param_space()

    assert "min_trend_period_hours" in spaces
    assert "atr_period" in spaces
    assert "k_stop" in spaces

    assert spaces["atr_period"].param_type == "int"
    assert spaces["k_stop"].param_type == "float"


def test_grid_search():
    """Test grid search candidate generation."""
    spaces = {
        "param1": ParamSpace("param1", "int", min_val=1, max_val=3),
        "param2": ParamSpace("param2", "float", min_val=0.0, max_val=1.0),
    }

    search = GridSearch(spaces, n_points_per_param=3, seed=42)
    candidates = search.generate_candidates()

    assert len(candidates) == 9  # 3 x 3
    assert all("param1" in c and "param2" in c for c in candidates)


def test_random_search():
    """Test random search candidate generation."""
    spaces = {
        "param1": ParamSpace("param1", "int", min_val=1, max_val=100),
        "param2": ParamSpace("param2", "float", min_val=0.0, max_val=1.0),
    }

    search = RandomSearch(spaces, n_iter=50, seed=42)
    candidates = search.generate_candidates()

    assert len(candidates) == 50
    assert all("param1" in c and "param2" in c for c in candidates)

    # Check ranges
    for c in candidates:
        assert 1 <= c["param1"] <= 100
        assert 0.0 <= c["param2"] <= 1.0


def test_random_search_log_scale():
    """Test random search with log scale."""
    spaces = {
        "param1": ParamSpace("param1", "float", min_val=0.001, max_val=1.0, log_scale=True),
    }

    search = RandomSearch(spaces, n_iter=100, seed=42)
    candidates = search.generate_candidates()

    values = [c["param1"] for c in candidates]
    log_values = [np.log10(v) for v in values]

    # Log-scale should produce more uniform distribution in log space
    assert min(values) >= 0.001
    assert max(values) <= 1.0


def test_random_search_categorical():
    """Test random search with categorical parameters."""
    spaces = {
        "method": ParamSpace("method", "categorical", categories=["A", "B", "C"]),
    }

    search = RandomSearch(spaces, n_iter=30, seed=42)
    candidates = search.generate_candidates()

    methods = [c["method"] for c in candidates]
    assert all(m in ["A", "B", "C"] for m in methods)


def test_bayesian_search_initialization():
    """Test Bayesian search initialization."""
    spaces = {
        "param1": ParamSpace("param1", "float", min_val=0.0, max_val=1.0),
    }

    search = BayesianSearch(spaces, n_initial=5, n_iter=10, seed=42)

    # Should generate initial random samples
    candidates = search.generate_candidates()
    assert len(candidates) == 5


def test_bayesian_search_update():
    """Test Bayesian search observation updates."""
    spaces = {
        "param1": ParamSpace("param1", "float", min_val=0.0, max_val=1.0),
    }

    search = BayesianSearch(spaces, n_initial=3, n_iter=5, seed=42)

    # Initial samples
    candidates = search.generate_candidates()
    for i, c in enumerate(candidates):
        search.update_observations(c, score=float(i))

    assert len(search.X_observed) == 3
    assert len(search.y_observed) == 3

    # BO iterations
    bo_candidates = search.generate_candidates()
    assert len(bo_candidates) == 5


def test_walkforward_windows_rolling():
    """Test rolling walk-forward window creation."""
    n_bars = 1000
    train_size = 300
    test_size = 100

    windows = create_walkforward_windows(n_bars, train_size, test_size)

    assert len(windows) > 0
    for window in windows:
        assert window.train_end_idx - window.train_start_idx == train_size
        assert window.test_end_idx - window.test_start_idx == test_size
        assert window.train_end_idx == window.test_start_idx


def test_walkforward_windows_anchored():
    """Test anchored walk-forward window creation."""
    n_bars = 1000
    train_size = 300
    test_size = 100

    windows = create_walkforward_windows(n_bars, train_size, test_size, anchored=True)

    assert len(windows) > 0
    for window in windows:
        assert window.train_start_idx == 0  # Anchored
        assert window.test_end_idx - window.test_start_idx == test_size


def test_split_data_by_window():
    """Test data splitting by window."""
    df = pd.DataFrame({"value": range(100)})
    window = WalkForwardWindow(
        train_start_idx=0,
        train_end_idx=60,
        test_start_idx=60,
        test_end_idx=80,
        window_id=0,
    )

    train_df, test_df = split_data_by_window(df, window)

    assert len(train_df) == 60
    assert len(test_df) == 20
    assert train_df["value"].iloc[0] == 0
    assert train_df["value"].iloc[-1] == 59
    assert test_df["value"].iloc[0] == 60


def test_block_bootstrap_returns():
    """Test block bootstrap for returns."""
    returns = np.random.randn(100) * 0.01

    bootstrap_samples = block_bootstrap_returns(returns, n_simulations=10, block_size=5, seed=42)

    assert len(bootstrap_samples) == 10
    for sample in bootstrap_samples:
        assert len(sample) == len(returns)


def test_monte_carlo_equity_curves():
    """Test Monte Carlo equity curve generation."""
    equity_curve = np.array([10000.0 * (1.001 ** i) for i in range(100)])

    mc_curves = monte_carlo_equity_curves(equity_curve, n_simulations=10, block_size=5, seed=42)

    assert len(mc_curves) == 10
    for curve in mc_curves:
        assert len(curve) == len(equity_curve)
        assert curve[0] == equity_curve[0]  # Same starting point


def test_compute_mc_metrics():
    """Test Monte Carlo metrics computation."""
    equity_curves = [
        np.array([10000.0 * (1.001 ** i) for i in range(100)]) for _ in range(10)
    ]

    mc_result = compute_mc_metrics(equity_curves, initial_capital=10000.0)

    assert mc_result.n_simulations == 10
    assert "sharpe_ratio" in mc_result.metrics_distributions
    assert "total_return" in mc_result.mean_metrics
    assert "p50" in mc_result.percentiles["sharpe_ratio"]


def test_compute_combined_oos_metrics():
    """Test combined OOS metrics computation."""
    equity_curves = [
        np.array([10000.0, 10100.0, 10200.0]),
        np.array([10200.0, 10250.0, 10300.0]),
    ]

    metrics = compute_combined_oos_metrics(equity_curves)

    assert "total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown_pct" in metrics


def test_optimization_runner_objective():
    """Test optimization runner with simple objective function."""

    def simple_objective(params: StrategyParams, df: pd.DataFrame) -> dict[str, float]:
        # Simple quadratic objective
        x = params.k_stop - 2.0
        score = -(x**2)
        return {"score": score, "k_stop": params.k_stop}

    df = pd.DataFrame({"close": np.random.randn(100)})

    runner = OptimizationRunner(
        objective_function=simple_objective,
        objective_metric="score",
        maximize=True,
        seed=42,
    )

    spaces = {
        "k_stop": ParamSpace("k_stop", "float", min_val=0.5, max_val=4.0),
    }

    opt_run = runner.run_random_search(spaces, df, n_iter=20, verbose=False)

    assert len(opt_run.results) == 20
    assert opt_run.best_params is not None
    assert 0.5 <= opt_run.best_params.k_stop <= 4.0


def test_optimization_result_dataclass():
    """Test OptimizationResult dataclass."""
    params = StrategyParams(k_stop=2.0)
    result = OptimizationResult(
        params=params,
        train_metrics={"sharpe": 1.5},
        test_metrics=None,
        runtime_seconds=0.5,
        param_id=0,
    )

    assert result.params.k_stop == 2.0
    assert result.train_metrics["sharpe"] == 1.5
    assert result.runtime_seconds == 0.5
