# M8 Implementation: Parameter Optimization and Robustness

## Overview

M8 adds comprehensive parameter optimization and robustness evaluation capabilities, including Grid/Random/Bayesian search, walk-forward analysis, Monte Carlo resampling, and rich visualizations.

## Features Implemented

### 1. Parameter Search Methods

**Module**: `core/optimization/search.py`

Three optimization algorithms are provided:

#### Grid Search
Exhaustive search over parameter grid with configurable granularity.

```python
from core.optimization.search import GridSearch
from core.optimization.params import create_default_param_space

spaces = create_default_param_space()
search = GridSearch(spaces, n_points_per_param=5, seed=42)
candidates = search.generate_candidates()  # Returns all grid combinations
```

#### Random Search
Random sampling from parameter space with support for log-scale parameters.

```python
from core.optimization.search import RandomSearch

search = RandomSearch(spaces, n_iter=100, seed=42)
candidates = search.generate_candidates()  # Returns 100 random samples
```

#### Bayesian Optimization
Gaussian Process-based optimization with acquisition functions (EI, UCB, POI).

```python
from core.optimization.search import BayesianSearch

search = BayesianSearch(
    spaces,
    n_initial=10,      # Initial random samples
    n_iter=50,         # BO iterations
    seed=42,
    acquisition="ei",  # Expected Improvement
    xi=0.01,
)

# Initial samples
candidates = search.generate_candidates()
for params in candidates:
    score = evaluate(params)
    search.update_observations(params, score)

# BO iterations
bo_candidates = search.generate_candidates()
for params in bo_candidates:
    score = evaluate(params)
    search.update_observations(params, score)
```

### 2. Parameter Space Definition

**Module**: `core/optimization/params.py`

Define searchable parameter spaces with type safety:

```python
from core.optimization.params import ParamSpace, StrategyParams

# Integer parameter
atr_period = ParamSpace(
    name="atr_period",
    param_type="int",
    min_val=7,
    max_val=28,
)

# Float parameter with log scale
cutoff_scale = ParamSpace(
    name="cutoff_scale",
    param_type="float",
    min_val=0.5,
    max_val=2.0,
    log_scale=False,
)

# Categorical parameter
method = ParamSpace(
    name="method",
    param_type="categorical",
    categories=["ATR", "Residual"],
)
```

**Strategy Parameters**:
- `min_trend_period_hours`: Trend smoothing period
- `cutoff_scale`: DCT smoothing aggressiveness
- `atr_period`: ATR calculation period
- `k_stop`: Stop loss multiplier
- `k_profit`: Take profit multiplier
- `slope_threshold`: Minimum trend slope
- `slope_lookback`: Slope calculation lookback

### 3. Walk-Forward Analysis

**Module**: `core/optimization/walkforward.py`

Rolling or anchored walk-forward validation:

```python
from core.optimization.walkforward import create_walkforward_windows

# Rolling windows
windows = create_walkforward_windows(
    n_bars=1000,
    train_size=300,    # Train on 300 bars
    test_size=100,     # Test on 100 bars
    step_size=100,     # Roll forward 100 bars
    anchored=False,    # Rolling window
)

# Anchored windows (growing training set)
windows = create_walkforward_windows(
    n_bars=1000,
    train_size=300,
    test_size=100,
    anchored=True,     # Training set grows each window
)
```

Each window optimizes parameters on training data and validates on out-of-sample test data.

### 4. Monte Carlo Resampling

**Module**: `core/optimization/monte_carlo.py`

Block bootstrap for robustness testing:

```python
from core.optimization.monte_carlo import (
    monte_carlo_equity_curves,
    compute_mc_metrics,
)

# Generate bootstrap samples
mc_curves = monte_carlo_equity_curves(
    equity_curve,
    n_simulations=1000,
    block_size=24,      # 24-hour blocks
    seed=42,
)

# Compute metrics distributions
mc_result = compute_mc_metrics(mc_curves, initial_capital=10000.0)

print(f"Mean Sharpe: {mc_result.mean_metrics['sharpe_ratio']:.3f}")
print(f"5th percentile: {mc_result.percentiles['sharpe_ratio']['p5']:.3f}")
print(f"95th percentile: {mc_result.percentiles['sharpe_ratio']['p95']:.3f}")
```

Methods available:
- `block_bootstrap_returns()`: Bootstrap returns with block structure
- `block_bootstrap_residuals()`: Bootstrap residuals for synthetic price generation
- `monte_carlo_equity_curves()`: Generate synthetic equity curves
- `compute_mc_metrics()`: Compute metric distributions and percentiles

### 5. Optimization Runner

**Module**: `core/optimization/runner.py`

Unified interface for running optimizations:

```python
from core.optimization.runner import OptimizationRunner

def objective_function(params, df):
    # Run backtest and return metrics
    result = run_strategy(params, df)
    return result.metrics

runner = OptimizationRunner(
    objective_function=objective_function,
    objective_metric="sharpe_ratio",
    maximize=True,
    seed=42,
)

# Run random search
opt_run = runner.run_random_search(
    param_spaces=spaces,
    data=df,
    n_iter=100,
    verbose=True,
)

# Run Bayesian optimization
opt_run = runner.run_bayesian_search(
    param_spaces=spaces,
    data=df,
    n_initial=10,
    n_iter=40,
    verbose=True,
)

# Run walk-forward analysis
opt_run, wf_result = runner.run_walkforward(
    param_spaces=spaces,
    data=df,
    train_size=500,
    test_size=200,
    search_method="random",
    n_candidates=50,
    verbose=True,
)
```

Results include:
- Leaderboard with all evaluated configurations
- Best parameters and score
- Runtime statistics
- Train/test metrics per configuration

### 6. Visualization

**Module**: `core/optimization/visualization.py`

Rich visualizations for analysis:

```python
from core.optimization.visualization import (
    plot_param_heatmap,
    plot_frontier,
    plot_optimization_progress,
    plot_parameter_importance,
    plot_walkforward_results,
    plot_monte_carlo_distribution,
)

# Parameter heatmap
fig = plot_param_heatmap(
    leaderboard=opt_run.leaderboard,
    param_x="k_stop",
    param_y="k_profit",
    metric="train_sharpe_ratio",
)

# Efficient frontier (risk vs return)
fig = plot_frontier(
    leaderboard=opt_run.leaderboard,
    metric_x="train_sharpe_ratio",
    metric_y="train_max_drawdown_pct",
)

# Walk-forward results
fig = plot_walkforward_results(wf_result, metric="sharpe_ratio")

# Monte Carlo distribution
fig = plot_monte_carlo_distribution(mc_result, metric="sharpe_ratio")

# Parameter importance
fig = plot_parameter_importance(
    leaderboard=opt_run.leaderboard,
    metric="train_sharpe_ratio",
    top_n=10,
)

# Optimization progress
fig = plot_optimization_progress(opt_run)
```

### 7. Export Utilities

**Module**: `core/optimization/export.py`

Export results to various formats:

```python
from core.optimization.export import (
    export_leaderboard_csv,
    export_leaderboard_parquet,
    export_best_config_json,
    export_walkforward_results,
    export_monte_carlo_results,
    export_full_optimization_results,
)

# Export leaderboard
export_leaderboard_csv(opt_run, "results/leaderboard.csv")
export_leaderboard_parquet(opt_run, "results/leaderboard.parquet")

# Export best configuration
export_best_config_json(opt_run, "results/best_config.json")

# Export walk-forward results
export_walkforward_results(wf_result, "results/walkforward/")

# Export Monte Carlo results
export_monte_carlo_results(mc_result, "results/monte_carlo/")

# Export everything with visualizations
export_full_optimization_results(
    opt_run,
    "results/full/",
    include_visualizations=True,
)
```

### 8. Streamlit UI Integration

**Module**: `app/ui/optimization_tab.py`

Interactive optimization interface:

Features:
- Select data range and symbols
- Choose search method (Grid/Random/Bayesian)
- Configure walk-forward analysis
- Enable Monte Carlo resampling
- View leaderboard and best configs
- Interactive visualizations
- Export results

Access via the "🔍 Optimization" tab in the Streamlit UI.

## Complete Example

See `examples/optimization_example.py` for a comprehensive example:

```bash
python examples/optimization_example.py
```

This example demonstrates:
1. Loading data
2. Running random search (100 iterations)
3. Running Bayesian optimization (10+40 iterations)
4. Walk-forward analysis
5. Monte Carlo resampling
6. Generating visualizations
7. Exporting results

## Optimization Workflow

### 1. Exploratory Phase
Start with random search to explore the parameter space:

```python
opt_run = runner.run_random_search(spaces, df, n_iter=100)
```

### 2. Refinement Phase
Use Bayesian optimization to refine around promising regions:

```python
opt_run = runner.run_bayesian_search(spaces, df, n_initial=20, n_iter=50)
```

### 3. Validation Phase
Test robustness with walk-forward analysis:

```python
opt_run, wf_result = runner.run_walkforward(
    spaces, df, train_size=500, test_size=200, n_candidates=30
)
```

### 4. Robustness Testing
Evaluate stability with Monte Carlo:

```python
mc_curves = monte_carlo_equity_curves(equity_curve, n_simulations=1000)
mc_result = compute_mc_metrics(mc_curves)
```

## Performance Considerations

### Grid Search
- Time complexity: O(k^n) where k = points per param, n = num params
- Use for small parameter spaces (2-3 params with 3-5 points each)
- Exhaustive but slow for large spaces

### Random Search
- Time complexity: O(m) where m = num iterations
- Good for initial exploration
- Use 50-200 iterations for reasonable coverage

### Bayesian Optimization
- Time complexity: O(m) but with overhead per iteration
- Most efficient for expensive objective functions
- Use 10-20 initial samples + 30-50 BO iterations
- Best for fine-tuning around good regions

### Walk-Forward
- Runs full optimization per window
- Time = num_windows × optimization_time
- Use fewer candidates per window (20-50)
- Consider anchored mode for faster convergence

### Monte Carlo
- Time = num_simulations × metric_computation
- 1000-5000 simulations recommended
- Block size: ~24 hours for intraday data
- Parallelizable (not implemented yet)

## Reproducibility

All methods support random seeds:

```python
runner = OptimizationRunner(
    objective_function=objective_fn,
    objective_metric="sharpe_ratio",
    maximize=True,
    seed=42,  # Reproducible results
)
```

Seeds control:
- Random search sampling
- Bayesian optimization random components
- Bootstrap resampling
- Walk-forward train/test splits (if randomized)

## Metrics Available

All standard backtest metrics are available for optimization:
- `sharpe_ratio`: Risk-adjusted return
- `sortino_ratio`: Downside risk-adjusted return
- `total_return`: Total P&L percentage
- `annualized_return`: Annualized return
- `max_drawdown_pct`: Maximum drawdown percentage
- `profit_factor`: Gross profit / Gross loss
- `win_rate`: Percentage of winning trades
- `n_trades`: Number of trades
- `avg_bars_held`: Average trade duration

## Testing

Comprehensive test coverage:

```bash
pytest tests/test_optimization.py -v
```

Tests cover:
- Parameter space definitions
- Search algorithm correctness
- Walk-forward window generation
- Monte Carlo bootstrap methods
- Optimization runner integration
- Export utilities

## Future Enhancements

Potential additions:
- Parallel optimization (multi-processing)
- Multi-objective optimization (Pareto frontiers)
- Constraint handling (e.g., max trades, max leverage)
- Online optimization (adaptive parameters)
- Ensemble methods (combining multiple configs)
- Cross-validation integration
- GPU acceleration for large-scale searches
- Advanced acquisition functions
- Hyperparameter tuning for search algorithms

## Configuration Best Practices

### Parameter Ranges

Set realistic bounds based on domain knowledge:

```python
# Trend period: 12 hours to 1 week
min_trend_period_hours: 12.0 - 168.0

# Cutoff scale: moderate to aggressive smoothing
cutoff_scale: 0.5 - 2.0

# ATR period: 1-4 weeks
atr_period: 7 - 28

# Stop loss: 1-4 ATR
k_stop: 1.0 - 4.0

# Take profit: 2-6 ATR
k_profit: 2.0 - 6.0
```

### Objective Selection

Choose metric based on trading goals:
- **Sharpe Ratio**: Best for risk-adjusted performance
- **Sortino Ratio**: Focus on downside risk
- **Total Return**: Pure profit maximization
- **Profit Factor**: Emphasize win/loss ratio

### Walk-Forward Settings

Balance between validation rigor and computation:
- Train size: 30-60% of data
- Test size: 10-20% of data
- Step size: Equal to test size (no overlap)
- Anchored: Use for limited data or regime-dependent strategies

### Monte Carlo Settings

Balance between statistical power and runtime:
- Simulations: 1000-5000 for stable estimates
- Block size: Match typical trend duration
  - 24 hours for intraday strategies
  - 7 days for swing strategies

## UI Usage

1. Navigate to "🔍 Optimization" tab
2. Configure data range and symbol
3. Select search method
4. (Optional) Enable walk-forward analysis
5. (Optional) Enable Monte Carlo resampling
6. Click "🚀 Run Optimization"
7. View results in tabs:
   - **Leaderboard**: All configurations ranked
   - **Best Config**: Optimal parameters
   - **Visualizations**: Heatmaps, frontiers, progress
   - **Walk-Forward**: OOS validation results
   - **Monte Carlo**: Robustness statistics
8. Export results to CSV/Parquet

## License

See project LICENSE file.
