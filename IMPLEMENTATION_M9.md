# M9 Implementation: Portfolio and Risk Management

## Overview

M9 adds comprehensive multi-symbol portfolio management and risk control capabilities, including parallel backtesting, multiple weighting schemes, correlation analysis, and portfolio-level metrics.

## Features Implemented

### 1. Portfolio Weighting Schemes

**Module**: `core/portfolio/weights.py`

Multiple weighting methodologies are provided for portfolio construction:

#### Equal Weights
Simplest approach - equal allocation to all assets.

```python
from core.portfolio.weights import compute_equal_weights

weights = compute_equal_weights(n_assets=3)
# Returns: [0.333, 0.333, 0.334]
```

#### Volatility-Scaled Weights
Inverse volatility weighting - lower volatility assets receive higher weights.

```python
from core.portfolio.weights import compute_volatility_scaled_weights

weights = compute_volatility_scaled_weights(
    returns=returns_df,
    target_vol=0.02,
    lookback=60,
    min_weight=0.0,
    max_weight=1.0,
)
```

Features:
- Scales weights inversely to volatility
- Applies minimum/maximum weight constraints
- Automatically normalizes to sum to 1.0

#### Risk Parity Weights
Approximately equal risk contribution from each asset.

```python
from core.portfolio.weights import compute_risk_parity_weights

weights = compute_risk_parity_weights(
    returns=returns_df,
    lookback=60,
    max_iterations=100,
    tolerance=1e-6,
    min_weight=0.0,
    max_weight=1.0,
)
```

Features:
- Iterative algorithm for risk parity
- Each asset contributes equally to portfolio risk
- Supports weight constraints

#### Market Cap Weights
Weight assets by market capitalization.

```python
from core.portfolio.weights import compute_market_cap_weights

weights = compute_market_cap_weights(
    market_caps={"BTC": 1000.0, "ETH": 500.0},
    symbols=["BTC", "ETH"],
    min_weight=0.0,
    max_weight=1.0,
)
```

#### Weight Constraints and Rebalancing

Apply and enforce weight constraints:

```python
from core.portfolio.weights import apply_weight_caps, rebalance_weights

# Apply caps
capped = apply_weight_caps(weights, min_weight=0.1, max_weight=0.5)

# Check if rebalancing is needed
new_weights, should_rebalance = rebalance_weights(
    current_weights=current,
    target_weights=target,
    threshold=0.05,  # 5% deviation threshold
)
```

### 2. Portfolio Analytics

**Module**: `core/portfolio/analytics.py`

Comprehensive analytics for portfolio risk and correlation analysis.

#### Correlation Analysis

```python
from core.portfolio.analytics import (
    compute_correlation_matrix,
    compute_rolling_correlation,
)

# Static correlation matrix
corr_matrix = compute_correlation_matrix(
    returns=returns_df,
    method="pearson",  # or "spearman", "kendall"
    min_periods=30,
)

# Rolling pairwise correlations
rolling_corr = compute_rolling_correlation(
    returns=returns_df,
    window=60,
    method="pearson",
)
# Returns: dict[(symbol1, symbol2)] -> pd.Series
```

#### Portfolio Risk Metrics

```python
from core.portfolio.analytics import (
    compute_portfolio_volatility,
    compute_diversification_ratio,
    compute_risk_contributions,
)

# Portfolio volatility
port_vol = compute_portfolio_volatility(weights, cov_matrix)

# Diversification ratio (higher = better diversification)
div_ratio = compute_diversification_ratio(
    weights=weights,
    volatilities=asset_vols,
    portfolio_vol=port_vol,
)

# Risk contribution by asset
risk_contrib = compute_risk_contributions(weights, cov_matrix)
```

#### Concentration Metrics

```python
from core.portfolio.analytics import compute_concentration_metrics

metrics = compute_concentration_metrics(weights)

# Returns:
# - herfindahl_index: Sum of squared weights
# - effective_n: Effective number of assets (1/HHI)
# - max_weight: Maximum single asset weight
# - top3_concentration: Sum of top 3 weights
```

#### Beta and Tracking Error

```python
from core.portfolio.analytics import compute_beta, compute_tracking_error

# Compute beta relative to market
beta = compute_beta(asset_returns, market_returns)

# Tracking error relative to benchmark
te = compute_tracking_error(portfolio_returns, benchmark_returns)
```

#### Sector Exposure

```python
from core.portfolio.analytics import compute_exposure_by_sector

exposure = compute_exposure_by_sector(
    weights=weights,
    symbols=["BTC", "ETH", "SOL"],
    sector_map={"BTC": "Layer1", "ETH": "Layer1", "SOL": "Layer1"},
)
# Returns: {"Layer1": 1.0}
```

### 3. Portfolio Executor

**Module**: `core/portfolio/executor.py`

Manages parallel execution of per-symbol backtests and result aggregation.

#### Single Symbol Backtest

```python
from core.portfolio.executor import run_single_symbol_backtest

result = run_single_symbol_backtest(
    symbol="BTCUSDT",
    df=ohlcv_df,
    strategy_func=my_strategy,
    strategy_params={"param1": value1},
    config=backtest_config,
)
```

Returns `SymbolBacktestResult` with:
- `symbol`: Symbol name
- `result`: BacktestResult
- `returns`: Return series
- `equity_curve`: Equity curve

#### Parallel Multi-Symbol Backtests

```python
from core.portfolio.executor import run_parallel_backtests

results = run_parallel_backtests(
    symbols=["BTCUSDT", "ETHUSDT"],
    data_dict={"BTCUSDT": df1, "ETHUSDT": df2},
    strategy_func=my_strategy,
    strategy_params=params,
    config=backtest_config,
    max_workers=4,  # Optional: parallel workers
)
# Returns: dict[symbol -> SymbolBacktestResult]
```

#### Result Alignment and Statistics

```python
from core.portfolio.executor import (
    align_symbol_results,
    compute_symbol_statistics,
)

# Align equity curves and returns
equity_df, returns_df = align_symbol_results(results)

# Compute summary statistics
stats_df = compute_symbol_statistics(results)
```

### 4. Portfolio Manager

**Module**: `core/portfolio/portfolio.py`

Main portfolio management interface that orchestrates weighting, backtesting, and analytics.

#### Portfolio Configuration

```python
from core.portfolio.portfolio import PortfolioConfig

portfolio_config = PortfolioConfig(
    weighting_method="equal",  # or "volatility", "risk_parity"
    rebalance_frequency=24,    # Hours between rebalances
    rebalance_threshold=0.05,  # 5% deviation triggers rebalance
    min_weight=0.0,
    max_weight=1.0,
    target_vol=0.02,           # For volatility scaling
    lookback_period=60,        # Bars for vol/cov calculation
)
```

#### Creating a Portfolio

```python
from core.portfolio.portfolio import create_portfolio
from core.backtest.engine import BacktestConfig

portfolio = create_portfolio(
    symbols=["BTCUSDT", "ETHUSDT"],
    weighting_method="risk_parity",
    initial_capital=10000.0,
)
```

#### Running Portfolio Backtest

```python
# Define strategy function
def my_strategy(df, **params):
    # Generate signals
    return signals

# Run portfolio backtest
result = portfolio.run_backtest(
    data_dict={"BTCUSDT": df1, "ETHUSDT": df2},
    strategy_func=my_strategy,
    strategy_params={"param1": value1},
)
```

#### Portfolio Result

`PortfolioResult` contains:
- `equity_curve`: Portfolio equity over time
- `weights`: Current portfolio weights
- `symbols`: List of symbols
- `symbol_results`: Per-symbol backtest results
- `returns`: DataFrame of asset returns
- `portfolio_returns`: Portfolio return series
- `metrics`: Comprehensive portfolio metrics
- `correlation_matrix`: Asset correlation matrix
- `rebalance_dates`: List of rebalance timestamps

Portfolio metrics include:
- Total return, annualized volatility
- Sharpe ratio, max drawdown
- Herfindahl index, effective N assets
- Diversification ratio
- Number of rebalances

### 5. Streamlit UI Integration

**Module**: `app/ui/portfolio_tab.py`

Interactive portfolio management interface in Streamlit.

Access via the "📊 Portfolio" mode in the sidebar.

#### Features:

1. **Symbol Selection**
   - Multi-select from supported symbols
   - Requires minimum 2 symbols

2. **Weighting Configuration**
   - Choose weighting method (equal, volatility, risk parity)
   - Set rebalancing frequency
   - Configure weight constraints

3. **Strategy Parameters**
   - Trend period, cutoff scale
   - ATR parameters, stop loss multipliers

4. **Visualizations**
   - Portfolio equity curve with rebalance markers
   - Current weight distribution (table + pie chart)
   - Correlation heatmap
   - Individual symbol performance comparison
   - Normalized returns comparison

5. **Metrics Display**
   - Total return, Sharpe ratio, max drawdown
   - Diversification ratio, effective N assets
   - Concentration metrics (HHI, top 3 concentration)
   - Per-symbol statistics table

## Complete Example

See `examples/portfolio_example.py` for a comprehensive example:

```bash
python examples/portfolio_example.py
```

This example demonstrates:
1. Loading data for multiple symbols
2. Running backtests with different weighting schemes
3. Comparing portfolio performance
4. Analyzing correlations and risk metrics

## Usage Patterns

### Basic Portfolio Backtest

```python
from datetime import UTC, datetime
from core.data.loader import load_klines
from core.portfolio.portfolio import create_portfolio

# Load data
symbols = ["BTCUSDT", "ETHUSDT"]
data_dict = {}
for symbol in symbols:
    df = load_klines(
        symbol=symbol,
        interval="1h",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 6, 1, tzinfo=UTC),
    )
    data_dict[symbol] = df

# Create portfolio
portfolio = create_portfolio(
    symbols=symbols,
    weighting_method="equal",
    initial_capital=10000.0,
)

# Run backtest
result = portfolio.run_backtest(
    data_dict=data_dict,
    strategy_func=my_strategy,
    strategy_params={},
)

# Analyze results
print(f"Total Return: {result.metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Diversification: {result.metrics['diversification_ratio']:.2f}")
```

### Comparing Weighting Schemes

```python
weighting_methods = ["equal", "volatility", "risk_parity"]

for method in weighting_methods:
    portfolio = create_portfolio(
        symbols=symbols,
        weighting_method=method,
    )
    
    result = portfolio.run_backtest(
        data_dict=data_dict,
        strategy_func=my_strategy,
        strategy_params={},
    )
    
    print(f"\n{method.upper()}:")
    print(f"  Return: {result.metrics['total_return']:.2%}")
    print(f"  Sharpe: {result.metrics['sharpe_ratio']:.2f}")
    print(f"  Weights: {result.weights}")
```

### Advanced Portfolio with Constraints

```python
from core.portfolio.portfolio import Portfolio, PortfolioConfig
from core.backtest.engine import BacktestConfig

portfolio_config = PortfolioConfig(
    weighting_method="risk_parity",
    rebalance_frequency=24,
    rebalance_threshold=0.05,
    min_weight=0.2,  # Minimum 20% per asset
    max_weight=0.6,  # Maximum 60% per asset
    lookback_period=90,
)

backtest_config = BacktestConfig(
    initial_capital=10000.0,
    fee_rate=0.001,
    slippage=0.0005,
)

portfolio = Portfolio(
    symbols=symbols,
    portfolio_config=portfolio_config,
    backtest_config=backtest_config,
)

result = portfolio.run_backtest(
    data_dict=data_dict,
    strategy_func=my_strategy,
    strategy_params={},
)
```

## Testing

Comprehensive test coverage:

```bash
# Run all portfolio tests
pytest tests/test_portfolio*.py -v

# Run specific test modules
pytest tests/test_portfolio_weights.py -v
pytest tests/test_portfolio_analytics.py -v
pytest tests/test_portfolio.py -v
```

Tests cover:
- Weighting scheme calculations
- Correlation and risk analytics
- Portfolio construction and backtesting
- Result aggregation and alignment
- Edge cases and error handling

## Performance Considerations

### Parallel Execution
- Currently runs sequentially due to function serialization
- Future enhancement: implement proper parallel execution
- Consider processing large portfolios in batches

### Rebalancing
- More frequent rebalancing increases turnover costs
- Set appropriate `rebalance_threshold` to avoid excessive trading
- Typical frequencies: daily (24h), weekly (168h)

### Lookback Periods
- Longer lookbacks provide more stable estimates
- Shorter lookbacks adapt faster to changing markets
- Balance between stability and responsiveness

## Best Practices

### Symbol Selection
- Choose symbols with sufficient liquidity
- Consider correlation when building portfolios
- Diversify across different sectors/asset classes

### Weighting Methods

**Equal Weights**:
- Best for: Simple diversification, similar assets
- Pros: Robust, no parameter estimation
- Cons: Ignores risk differences

**Volatility Scaling**:
- Best for: Risk-adjusted allocation
- Pros: Reduces exposure to volatile assets
- Cons: Sensitive to volatility estimation

**Risk Parity**:
- Best for: Equal risk contribution
- Pros: Balances risk across assets
- Cons: More complex, may concentrate in low-risk assets

### Rebalancing Strategy
- Daily (24h): Active management, higher costs
- Weekly (168h): Balanced approach
- Monthly: Passive, lower costs

Set `rebalance_threshold` to avoid unnecessary trades:
- 5%: Standard threshold
- 10%: Less frequent rebalancing
- 2%: More responsive

### Risk Management
- Apply weight caps to prevent over-concentration
- Monitor correlation changes over time
- Track diversification ratio (target > 1.2)
- Keep effective N assets high

## Future Enhancements

Potential additions:
- True parallel execution with multiprocessing
- Time-varying risk models (GARCH, etc.)
- Transaction cost modeling for rebalancing
- Dynamic weight optimization
- Multi-objective optimization (return vs risk)
- Constraint optimization (sector limits, etc.)
- Rolling portfolio optimization
- Monte Carlo portfolio simulation
- Conditional Value at Risk (CVaR) optimization
- Factor-based portfolio construction

## Integration with Existing Features

### With M8 Optimization
Optimize portfolio parameters:
```python
from core.optimization.runner import OptimizationRunner

def portfolio_objective(params, data_dict):
    portfolio = create_portfolio(symbols, **params)
    result = portfolio.run_backtest(data_dict, strategy, {})
    return result.metrics

runner = OptimizationRunner(portfolio_objective, "sharpe_ratio")
# Optimize rebalance frequency, weight caps, etc.
```

### With M7 MTF Features
Use MTF signals in portfolio:
```python
def mtf_strategy(df, **params):
    # Use MTF alignment for better signals
    from core.analysis.mtf import check_mtf_alignment
    # ... generate MTF-aligned signals
    return signals

result = portfolio.run_backtest(
    data_dict=data_dict,
    strategy_func=mtf_strategy,
    strategy_params=mtf_params,
)
```

## Documentation Updates

Updated files:
- `README.md`: Added portfolio features section
- `IMPLEMENTATION_M9.md`: This file
- `examples/portfolio_example.py`: Complete usage example

## API Summary

### Core Classes
- `Portfolio`: Main portfolio manager
- `PortfolioConfig`: Portfolio configuration
- `PortfolioResult`: Complete portfolio backtest results
- `SymbolBacktestResult`: Single symbol results

### Weight Functions
- `compute_equal_weights()`
- `compute_volatility_scaled_weights()`
- `compute_risk_parity_weights()`
- `compute_market_cap_weights()`
- `apply_weight_caps()`
- `rebalance_weights()`

### Analytics Functions
- `compute_correlation_matrix()`
- `compute_rolling_correlation()`
- `compute_portfolio_volatility()`
- `compute_diversification_ratio()`
- `compute_concentration_metrics()`
- `compute_risk_contributions()`
- `compute_beta()`
- `compute_tracking_error()`
- `compute_exposure_by_sector()`

### Executor Functions
- `run_single_symbol_backtest()`
- `run_parallel_backtests()`
- `align_symbol_results()`
- `compute_symbol_statistics()`
