# Architecture Deep Dive

[English](ARCHITECTURE.md) | [中文](ARCHITECTURE.zh-CN.md)

This document provides an in-depth technical explanation of the Binance Fourier Backtester's architecture, algorithms, and design decisions.

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Layer Architecture](#data-layer-architecture)
3. [Fourier Transform Methods](#fourier-transform-methods)
4. [Spectral Analysis](#spectral-analysis)
5. [Signal Generation Logic](#signal-generation-logic)
6. [Backtester Design](#backtester-design)
7. [Optimization Framework](#optimization-framework)
8. [Portfolio Management](#portfolio-management)
9. [Performance Considerations](#performance-considerations)

---

## System Overview

The platform uses a **layered architecture** with strict separation of concerns:

### Design Principles

1. **Separation of Concerns**: Each layer has a single, well-defined responsibility
2. **Dependency Inversion**: Core logic depends on abstractions, not implementations
3. **Type Safety**: Full type hints throughout for compile-time error detection
4. **Testability**: All components designed for unit testing with minimal mocking
5. **Performance**: Vectorized operations using NumPy for computational efficiency

### Layer Communication Flow

```
User Input → UI Layer → Core Analysis → Execution Engine → Data Layer
                ↓                           ↓
            Results ←─────── Metrics ←──── Storage
```

Data flows unidirectionally from data layer up through analysis to UI, with results flowing back down for storage/caching.

---

## Data Layer Architecture

### Binance REST API Client

**Location**: `core/data/binance_client.py`

#### Design Decisions

1. **httpx over requests**: Async support for future enhancements, better connection pooling
2. **tenacity for retries**: Declarative retry logic with exponential backoff
3. **Rate limiting**: Client-side tracking to avoid 429 errors

#### Retry Strategy

```python
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type((BinanceTransientError,)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def fetch_klines(...):
    # Fetch logic
```

**Rationale**: Binance API can have transient failures. Exponential backoff prevents thundering herd while allowing recovery from temporary issues.

#### Error Handling Hierarchy

```
BinanceError (base)
├── BinanceRateLimitError (429) → User should wait
├── BinanceRequestError (400, 404) → Invalid request, don't retry
└── BinanceTransientError (500, 503) → Retry with backoff
```

### Cache System

**Location**: `core/data/cache.py`

#### Why Parquet?

| Format | Read Speed | Write Speed | Compression | Schema |
|--------|------------|-------------|-------------|--------|
| CSV | Slow | Slow | Poor | No |
| JSON | Slow | Moderate | Poor | No |
| Pickle | Fast | Fast | Moderate | No |
| **Parquet** | **Very Fast** | **Fast** | **Excellent** | **Yes** |

**Parquet advantages**:
- Columnar format: Fast filtering on timestamp columns
- Built-in compression: ~10x smaller than CSV
- Schema preservation: Type safety without parsing
- pandas native support: Zero-copy reads

#### Gap Detection Algorithm

```python
def detect_gaps(df: pd.DataFrame, interval: str) -> list[tuple[datetime, datetime]]:
    """
    1. Compute expected interval duration (e.g., 1h = 3600s)
    2. Calculate diff between consecutive timestamps
    3. Flag gaps > 2x expected duration (allows for minor clock skew)
    4. Return list of (gap_start, gap_end) tuples
    """
```

**Why 2x threshold?** Allows for minor Binance API inconsistencies while catching real gaps.

#### Incremental Update Strategy

```
1. Load existing cache (if exists)
2. Find max timestamp in cache
3. Request data from max_timestamp + interval to present
4. Concatenate new data with cache
5. Drop duplicates (in case of overlap)
6. Sort by timestamp
7. Write back to Parquet
```

**Optimization**: Only fetches delta, not full historical dataset on each load.

---

## Fourier Transform Methods

### Why DCT over FFT?

| Method | Boundary Handling | Frequency Resolution | Real/Complex |
|--------|-------------------|---------------------|--------------|
| FFT | Periodic extension | High | Complex |
| **DCT** | **Mirror extension** | **High** | **Real** |
| DFT | Periodic extension | High | Complex |
| Wavelet | Depends on wavelet | Multi-resolution | Real |

**DCT advantages for financial data**:
1. **No periodicity assumption**: Prices aren't periodic, so FFT's periodic extension creates artifacts
2. **Real-valued**: Easier interpretation, no imaginary components
3. **Smooth boundaries**: Mirrored padding avoids edge discontinuities

### DCT Low-Pass Filtering Algorithm

**Location**: `core/analysis/fourier.py`

#### Mathematical Foundation

For a signal \( x[n] \) of length \( N \):

1. **Mirrored Padding**:
   ```
   Original:  [x₀, x₁, x₂, ..., xₙ]
   Padded:    [xₙ, ..., x₂, x₁, x₀, x₁, x₂, ..., xₙ, xₙ₋₁, ..., x₁]
   ```
   Ensures smooth transition at boundaries.

2. **DCT Transform**:
   ```
   X[k] = Σ x[n] * cos(πk(2n+1)/(2N))
   ```
   Converts to frequency domain.

3. **Low-Pass Filter**:
   ```
   H[k] = 1                           if k < k_cutoff
   H[k] = 0.5 * (1 + cos(π(k - k_cutoff) / taper_width))  if k_cutoff ≤ k < k_cutoff + taper
   H[k] = 0                           if k ≥ k_cutoff + taper
   ```
   Tapered cutoff prevents ringing (Gibbs phenomenon).

4. **Inverse DCT**:
   ```
   x_filtered[n] = IDCT(X[k] * H[k])
   ```

5. **Unpad**: Extract original length from center of filtered padded signal.

#### Cutoff Frequency Selection

Given desired minimum period `P_min` in bars:
```python
cutoff_freq = 1.0 / P_min
```

**Example**: For 24-hour trend on 1h data:
- `P_min = 24 bars`
- `cutoff_freq = 1/24 ≈ 0.042`

**Cutoff Scale**: User multiplier for fine-tuning:
- Scale = 0.5: Half the cutoff (more smoothing)
- Scale = 1.0: Exact cutoff
- Scale = 2.0: Double cutoff (less smoothing)

---

## Spectral Analysis

### FFT Power Spectrum

**Location**: `core/analysis/spectral.py`

#### Implementation

```python
def compute_fft_spectrum(signal, sampling_rate=1.0):
    """
    1. Apply Hanning window to reduce spectral leakage
    2. Compute FFT using scipy.fft.rfft (real FFT, half spectrum)
    3. Compute power: |X[k]|²
    4. Normalize by signal length
    5. Convert frequency bins to periods
    """
```

**Why Hanning window?** Reduces spectral leakage from finite signal length, provides better frequency resolution for non-periodic signals.

#### Dominant Peak Detection

```python
def find_dominant_peaks(freqs, power, min_distance=10):
    """
    1. Use scipy.signal.find_peaks with prominence threshold
    2. Sort peaks by power (descending)
    3. Return top N peaks with corresponding periods
    """
```

**min_distance parameter**: Prevents identifying harmonic peaks as separate cycles.

### Welch's Method (PSD)

#### Why Welch over Periodogram?

| Method | Variance | Frequency Resolution | Computational Cost |
|--------|----------|---------------------|-------------------|
| Periodogram | High | High | Low |
| **Welch** | **Low** | **Moderate** | **Moderate** |
| Multitaper | Very Low | Moderate | High |

**Welch advantages**:
- **Variance reduction**: Averages multiple windows
- **Good frequency resolution**: Overlapping windows preserve detail
- **Practical**: Fast enough for real-time UI

#### Welch Algorithm

```
1. Divide signal into overlapping segments
   - Window length: W (e.g., 256 bars)
   - Overlap: O = overlap_ratio * W (e.g., 50%)
   - Number of segments: (N - W) / (W - O) + 1

2. For each segment:
   a. Apply Hanning window
   b. Compute FFT
   c. Compute power spectrum

3. Average power spectra across all segments
```

**Parameter Selection**:
- **Window Length (W)**: 
  - Too small: Poor frequency resolution
  - Too large: Few segments, high variance
  - Rule of thumb: W = N/4 to N/8
- **Overlap**: 
  - 50% standard, 75% for smoother estimates

### Sliding Window Dominant Period

```python
def compute_sliding_dominant_period(signal, window_length=256, overlap_ratio=0.5):
    """
    1. Slide window across signal with overlap
    2. For each position:
       a. Extract window
       b. Compute Welch PSD
       c. Find dominant frequency (max power)
       d. Convert to period
    3. Return time series of dominant periods
    """
```

**Use Case**: Identifies regime changes (trending vs. mean-reverting).

**Interpretation**:
- Rising dominant period: Market moving to longer cycles (trending)
- Falling dominant period: Market moving to shorter cycles (choppy)
- Stable period: Consistent regime

---

## Signal Generation Logic

**Location**: `core/analysis/signals.py`

### Trend-Following Signals

#### Entry Logic

```python
def generate_entry_signal(close, smoothed, slope_threshold, slope_lookback):
    """
    Conditions for LONG entry:
    1. close[t] > smoothed[t]  (price above trend)
    2. slope(smoothed) > slope_threshold  (trend is rising)
    3. not currently in position
    
    Slope computed as:
    slope[t] = (smoothed[t] - smoothed[t - lookback]) / lookback
    """
```

**Design rationale**:
1. **Price above trend**: Ensures momentum confirmation
2. **Positive slope**: Avoids entries in declining trends
3. **Slope lookback**: Smooths slope to avoid noise-induced entries

#### Exit Logic

```python
def generate_exit_signal(close, smoothed, stop_levels):
    """
    Conditions for LONG exit:
    1. close[t] < smoothed[t]  (price below trend)
    OR
    2. close[t] < stop_levels[t]  (stop loss hit)
    
    Exit on next bar open after signal.
    """
```

**Why next-bar execution?** Prevents look-ahead bias. Signal generated at bar close, fill occurs at next bar open.

### Stop Loss Integration

Two stop methods available:

#### 1. ATR-Based Stops

```python
stop_loss = close - k_stop * ATR(period)
take_profit = close + k_profit * ATR(period)
```

**Advantages**:
- Adapts to volatility
- Wider stops in volatile periods (fewer whipsaws)
- Tighter stops in calm periods (protect profits)

**Parameter ranges**:
- `k_stop`: 1.5-3.0 (typically 2.0)
- `k_profit`: 2.0-4.0 (typically 3.0)
- `period`: 10-20 (typically 14)

#### 2. Residual-Based Stops

```python
residual = close - smoothed
sigma = rolling_std(residual, window)
stop_loss = smoothed - k_stop * sigma
take_profit = smoothed + k_profit * sigma
```

**Advantages**:
- Adapts to price-trend deviation
- Tighter in stable trends
- Wider during mean reversion

**When to use**:
- ATR: For breakout/momentum strategies
- Residual: For trend-following with tight control

---

## Backtester Design

**Location**: `core/backtest/engine.py`

### Vectorized vs. Event-Driven

| Approach | Speed | Flexibility | Complexity |
|----------|-------|-------------|------------|
| **Vectorized** | **Very Fast** | **Moderate** | **Low** |
| Event-Driven | Slow | High | High |

**Vectorized advantages**:
- 100-1000x faster for simple strategies
- Easier to debug (no state machine)
- Sufficient for trend-following without complex state

**Event-driven advantages**:
- Handles complex order types (limit, stop-limit)
- Portfolio rebalancing logic
- Real-time execution simulation

**Design choice**: Vectorized for speed, with enhancements for realistic fills.

### Backtest Algorithm

```python
def run_backtest(signals, open_prices, high, low, close, timestamps, config):
    """
    1. Initialize state
       - equity = initial_capital
       - position = 0
       - trades = []
    
    2. For each bar t:
       a. If signal[t] == 1 and position == 0:
          - entry_price = open[t+1]  (next bar open)
          - entry_price += slippage
          - position_size = compute_size(equity, volatility)
          - position = position_size / entry_price
          - equity -= position_size * (1 + fee_rate)
          
       b. If signal[t] == -1 and position > 0:
          - exit_price = open[t+1]
          - exit_price -= slippage
          - equity += position * exit_price * (1 - fee_rate)
          - record_trade(entry, exit)
          - position = 0
       
       c. If position > 0:
          - Check stop loss: if low[t+1] < stop[t], exit at stop
          - Check take profit: if high[t+1] > profit[t], exit at profit
       
       d. Update equity curve:
          - equity_curve[t] = cash + position * close[t]
    
    3. Compute metrics from equity curve and trades
    """
```

### Realistic Fill Modeling

#### 1. Next-Bar Fills

**Why not same-bar?** Prevents look-ahead bias. In reality, you can't trade at the close price where the signal is generated.

```python
# Signal at bar t close
if signal[t] == 1:
    # Fill at bar t+1 open
    entry_price = open[t+1]
```

#### 2. Stop/Profit Intrabar Checks

```python
# If position open and stop < high < profit
if low[t] <= stop <= high[t]:
    exit_price = stop  # Assume stop hit first
    
elif low[t] <= profit <= high[t]:
    exit_price = profit
```

**Assumption**: Conservative - stop assumed to hit before profit in same bar.

#### 3. Fees and Slippage

```python
# Entry
cost = quantity * entry_price * (1 + fee_rate) + slippage

# Exit  
proceeds = quantity * exit_price * (1 - fee_rate) - slippage
```

**Slippage modeling**: Fixed per-trade or percentage. More sophisticated models could use volume-based slippage.

### Position Sizing

**Location**: `core/analysis/sizing.py`

#### Volatility-Based Sizing

```python
def compute_volatility_size(equity, volatility_target, current_volatility):
    """
    Target volatility: Desired portfolio volatility (e.g., 2% per day)
    Current volatility: Asset's recent volatility (e.g., ATR or sigma)
    
    position_size = equity * (volatility_target / current_volatility)
    
    Normalizes risk across trades.
    """
```

**Example**:
- Equity: $10,000
- Target vol: 2%
- Asset vol: 4%
- Position size: $10,000 * (0.02 / 0.04) = $5,000 (50% of capital)

If volatility doubles, position size halves → constant risk.

#### Fixed Risk Sizing

```python
def compute_fixed_risk_size(equity, risk_per_trade, stop_distance):
    """
    risk_per_trade: Max $ to risk (e.g., 2% of equity)
    stop_distance: Distance to stop in %
    
    position_size = (equity * risk_per_trade) / stop_distance
    """
```

**Example**:
- Equity: $10,000
- Risk: 2% = $200
- Stop: 5% from entry
- Position size: $200 / 0.05 = $4,000

---

## Optimization Framework

**Location**: `core/optimization/`

### Search Algorithms

#### 1. Grid Search

```python
def grid_search(param_space, objective_func):
    """
    1. Generate Cartesian product of all parameter values
    2. For each combination:
       a. Run backtest
       b. Record objective metric
    3. Return sorted results
    
    Complexity: O(n^d) where n = values per param, d = dimensions
    """
```

**Use case**: Small parameter spaces (< 1000 combos), need exhaustive results.

**Advantages**:
- Guaranteed to find global optimum in grid
- Easy to visualize (2D heatmaps)

**Disadvantages**:
- Exponential growth with parameters
- Inefficient for continuous parameters

#### 2. Random Search

```python
def random_search(param_space, objective_func, n_iter):
    """
    1. For n_iter iterations:
       a. Sample random point from param_space
       b. Run backtest
       c. Record result
    2. Return sorted results
    
    Complexity: O(n_iter)
    """
```

**Use case**: Large parameter spaces, quick exploration.

**Advantages**:
- Linear complexity
- Often finds near-optimal in fewer iterations than grid
- Works well for high-dimensional spaces

**Disadvantages**:
- No guarantee of finding global optimum
- May miss narrow optimal regions

#### 3. Bayesian Optimization

```python
def bayesian_search(param_space, objective_func, n_initial, n_iter):
    """
    1. Initialize with n_initial random samples
    2. Fit Gaussian Process (GP) to results
    3. For n_iter iterations:
       a. Compute acquisition function (e.g., Expected Improvement)
       b. Find next point to sample (maximize acquisition)
       c. Evaluate objective at point
       d. Update GP
    4. Return best result
    
    Complexity: O(n³) per iteration (GP fitting)
    """
```

**Use case**: Expensive objective function, continuous parameters.

**Advantages**:
- Intelligent exploration-exploitation trade-off
- Converges faster for smooth objective surfaces
- Handles noisy objectives

**Disadvantages**:
- Slow for large n_iter (GP fitting)
- Less interpretable than grid
- Sensitive to initial samples

**Implementation**: Uses `scikit-optimize` with Gaussian Process regressor.

### Walk-Forward Analysis

**Location**: `core/optimization/walkforward.py`

#### Purpose

Validate that optimized parameters don't overfit to in-sample data.

#### Algorithm

```python
def walk_forward(data, param_space, train_window, test_window, mode="rolling"):
    """
    Rolling mode:
    1. Split data into overlapping train/test windows
       Train[0]: [0, train_window]
       Test[0]:  [train_window, train_window + test_window]
       Train[1]: [test_window, test_window + train_window]
       Test[1]:  [test_window + train_window, test_window*2 + train_window]
       ...
    
    For each fold:
    1. Optimize on train window
    2. Test on following test window
    3. Record out-of-sample performance
    
    4. Aggregate results:
       - Average OOS performance
       - Performance decay (in-sample vs. out-of-sample)
    """
```

**Anchored mode**: Train window grows (always starts at beginning), test window slides.

**Metrics**:
- **IS/OOS Ratio**: In-sample Sharpe / Out-of-sample Sharpe
  - < 1.5: Robust
  - 1.5-2.5: Moderate overfitting
  - \> 2.5: Severe overfitting
- **Degradation**: IS metric - OOS metric

### Monte Carlo Resampling

**Location**: `core/optimization/monte_carlo.py`

#### Block Bootstrap

```python
def monte_carlo_bootstrap(equity_curve, n_simulations, block_size):
    """
    1. Divide equity curve into blocks of size block_size
    2. For each simulation:
       a. Resample blocks with replacement
       b. Concatenate resampled blocks
       c. Compute performance metrics
    3. Analyze distribution of metrics across simulations
    
    Returns: percentile ranges (5th, 50th, 95th)
    """
```

**Why block bootstrap?** Preserves autocorrelation structure in returns (returns are not i.i.d.).

**Block size selection**:
- Too small: Breaks autocorrelation
- Too large: Reduces resampling diversity
- Rule of thumb: block_size = sqrt(N) or avg trade duration

**Use case**: Estimate confidence intervals for metrics, assess robustness to regime changes.

---

## Portfolio Management

**Location**: `core/portfolio/`

### Weighting Schemes

#### 1. Equal Weighting

```python
weights = [1/N] * N
```

**Use case**: Baseline, maximum diversification by count.

#### 2. Inverse Volatility

```python
vols = [volatility(symbol) for symbol in symbols]
inv_vols = [1/v for v in vols]
weights = [iv / sum(inv_vols) for iv in inv_vols]
```

**Effect**: Lower weight to high-volatility assets, equalizes risk contribution.

#### 3. Risk Parity

```python
def risk_parity_weights(cov_matrix):
    """
    Solve optimization:
    minimize Σ(RC_i - RC_j)²
    subject to Σw_i = 1, w_i ≥ 0
    
    where RC_i = w_i * (cov_matrix @ weights)[i]
    
    Uses scipy.optimize.minimize with Sequential Least Squares Programming (SLSQP).
    """
```

**Goal**: Equal marginal risk contribution from each asset.

**Use case**: Maximum risk diversification.

#### 4. Market Cap Weighted

```python
market_caps = get_market_caps(symbols)
weights = [mc / sum(market_caps) for mc in market_caps]
```

**Use case**: Mimic index, momentum bias (larger caps tend to have momentum).

### Rebalancing

```python
def rebalance(portfolio, target_weights, threshold=0.05):
    """
    1. Compute current weights from positions
    2. For each asset:
       if |current_weight - target_weight| > threshold:
           trade to adjust weight
    
    Rebalancing frequency: Daily, Weekly, Monthly, or threshold-based
    """
```

**Threshold-based**: Only rebalance if drift exceeds threshold (reduces transaction costs).

### Correlation Analytics

```python
def compute_diversification_ratio(weights, cov_matrix):
    """
    DR = (Σ w_i * σ_i) / σ_portfolio
    
    where:
    σ_i = std dev of asset i
    σ_portfolio = sqrt(weights^T @ cov_matrix @ weights)
    
    DR = 1: No diversification (perfect correlation)
    DR > 1: Diversification benefit
    DR = √N: Perfect diversification (zero correlation)
    """
```

**Interpretation**: Measures effectiveness of diversification.

---

## Performance Considerations

### Vectorization

**Key principle**: Replace Python loops with NumPy array operations.

**Example**:
```python
# Slow (Python loop)
result = []
for i in range(len(arr)):
    result.append(arr[i] * 2)

# Fast (vectorized)
result = arr * 2
```

**Speedup**: 10-100x for numerical operations.

**Where applied**:
- Signal generation
- Stop calculations
- Backtesting
- Metric computation

### Caching Strategy

**Memory hierarchy**:
```
Disk (Parquet) → Memory (DataFrame) → NumPy arrays → CPU cache
```

**Optimization**:
1. **Data loading**: Load once, cache in memory
2. **Column access**: Extract to NumPy for vectorization
3. **Intermediate results**: Cache smoothed prices, ATR, etc.

### Parallelization

**Location**: `core/portfolio/executor.py`

```python
from concurrent.futures import ProcessPoolExecutor

def parallel_backtest(symbols, strategy, params):
    """
    Uses multiprocessing to run independent backtests in parallel.
    Speedup: N_cores (CPU-bound workload)
    """
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(backtest, symbol, strategy, params) 
                   for symbol in symbols]
        results = [f.result() for f in futures]
    return results
```

**When to parallelize**:
- Portfolio backtests (independent symbols)
- Parameter optimization (independent combos)
- Monte Carlo simulations

**When not to**:
- Small workloads (overhead > benefit)
- Sequential dependencies

---

## Design Patterns Used

### 1. Strategy Pattern

**Location**: Signal generators, position sizers

Multiple implementations of same interface (e.g., ATR stops vs. residual stops).

### 2. Builder Pattern

**Location**: `BacktestConfig`, optimization parameter spaces

Fluent API for complex object construction.

### 3. Factory Pattern

**Location**: `create_portfolio()`, optimization runners

Encapsulates object creation logic.

### 4. Observer Pattern

**Location**: Live streaming coordinator

Observers (UI components) subscribe to data updates.

---

## Testing Strategy

### Unit Tests

- **Scope**: Individual functions
- **Mocking**: Minimal (prefer pure functions)
- **Coverage target**: > 80%

### Integration Tests

- **Scope**: End-to-end workflows (load data → backtest → metrics)
- **Real data**: Small datasets for fast execution

### Property-Based Tests

**Example**:
```python
def test_smoothed_price_bounded(close):
    """
    Property: smoothed price should be between min and max of original.
    """
    smoothed = smooth_price_series(close, min_period_bars=24)
    assert close.min() <= smoothed.min()
    assert smoothed.max() <= close.max()
```

---

## Further Reading

- **Fourier Transforms**: "Understanding Digital Signal Processing" by Richard Lyons
- **Backtesting**: "Quantitative Trading" by Ernest Chan
- **Portfolio Theory**: "Portfolio Selection" by Harry Markowitz
- **Optimization**: "Bayesian Optimization for Machine Learning" by Shahriari et al.
