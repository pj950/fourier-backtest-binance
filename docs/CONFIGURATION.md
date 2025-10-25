# Configuration Reference

[English](CONFIGURATION.md) | [中文](CONFIGURATION.zh-CN.md)

Complete reference for all configurable parameters in the Binance Fourier Backtester.

## Table of Contents

1. [Environment Variables](#environment-variables)
2. [Data Loading Parameters](#data-loading-parameters)
3. [Fourier Analysis Parameters](#fourier-analysis-parameters)
4. [Signal Generation Parameters](#signal-generation-parameters)
5. [Stop Loss Parameters](#stop-loss-parameters)
6. [Backtest Configuration](#backtest-configuration)
7. [Position Sizing Parameters](#position-sizing-parameters)
8. [Multi-Timeframe Parameters](#multi-timeframe-parameters)
9. [Optimization Parameters](#optimization-parameters)
10. [Portfolio Parameters](#portfolio-parameters)
11. [Presets and Defaults](#presets-and-defaults)

---

## Environment Variables

These are configured in the `.env` file.

### Data Paths

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BASE_PATH` | string | `./data` | Base directory for all data storage |
| `CACHE_DIR` | string | `./data/cache` | Directory for Parquet cache files |

**Example**:
```bash
BASE_PATH=/mnt/storage/trading_data
CACHE_DIR=/mnt/storage/trading_data/cache
```

### Binance API Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BINANCE_BASE_URL` | string | `https://api.binance.com` | Binance API endpoint |
| `BINANCE_API_KEY` | string | *(empty)* | Optional API key for higher rate limits |
| `BINANCE_API_SECRET` | string | *(empty)* | Optional API secret |
| `BINANCE_RATE_LIMIT_PER_MINUTE` | int | `1200` | Max requests per minute |
| `BINANCE_REQUEST_TIMEOUT` | int | `30` | Request timeout in seconds |

**Notes**:
- API key is optional for public market data
- Rate limits: 1200/min without key, higher with authenticated requests
- Use testnet URL `https://testnet.binance.vision` for testing

### Trading Defaults

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DEFAULT_FEE_RATE` | float | `0.001` | Default trading fee (0.1%) |
| `DEFAULT_SLIPPAGE_BPS` | float | `5.0` | Default slippage in basis points |

**Fee rates by venue**:
- Binance spot: 0.1% (0.001) regular, 0.075% with BNB
- Binance futures: 0.04% maker, 0.0675% taker
- FTX: 0.07% maker, 0.02% taker (before closure)

### Retry Settings

| Variable | Type | Default | Range | Description |
|----------|------|---------|-------|-------------|
| `MAX_RETRY_ATTEMPTS` | int | `5` | 1-10 | Number of retry attempts for failed requests |
| `RETRY_INITIAL_WAIT` | float | `1.0` | 0.5-5.0 | Initial wait time in seconds |
| `RETRY_MAX_WAIT` | float | `60.0` | 10-300 | Maximum wait time in seconds |

**Exponential backoff formula**:
```
wait_time = min(RETRY_INITIAL_WAIT * (2 ** attempt), RETRY_MAX_WAIT)
```

### UI Preset Storage

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PRESET_STORAGE_PATH` | string | `./data/presets/presets.yaml` | Path for saved parameter presets |
| `LAST_SESSION_STATE_PATH` | string | `./data/presets/last_state.yaml` | Path for last session state |

---

## Data Loading Parameters

### Symbol Selection

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `symbol` | string | BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, etc. | Trading pair to analyze |

**Supported symbols**: Defined in `core/data/loader.py` → `SUPPORTED_SYMBOLS`

**To add new symbols**: Edit `SUPPORTED_SYMBOLS` set in loader.py

### Interval Selection

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `interval` | string | 30m, 1h, 4h | Candlestick timeframe |

**Interval meanings**:
- `30m`: 30 minutes (48 candles per day)
- `1h`: 1 hour (24 candles per day)
- `4h`: 4 hours (6 candles per day)

**Considerations**:
- Lower timeframes: More data, higher noise, more signals
- Higher timeframes: Less data, smoother trends, fewer signals

### Date Range

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `start_date` | datetime | 2020-01-01 to present | Start of data range |
| `end_date` | datetime | `start_date` to present | End of data range |

**Recommendations**:
- **Testing**: 3-6 months for quick validation
- **Optimization**: 1-2 years for robust parameter search
- **Walk-forward**: 2-3 years minimum

**Historical availability**:
- Most pairs: Data available from 2020 or listing date
- Check Binance historical data availability per symbol

### Force Refresh

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force_refresh` | bool | `False` | Bypass cache and fetch fresh data |

**When to use**:
- ✅ Testing data pipeline changes
- ✅ Suspecting cache corruption
- ✅ Verifying latest data
- ❌ Normal usage (slow and wasteful)

---

## Fourier Analysis Parameters

### Min Trend Period

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `min_trend_hours` | float | 6-168 | `24.0` | Minimum trend period to preserve (hours) |

**Conversion to bars**:
```python
min_period_bars = min_trend_hours / interval_hours
```

**Examples** (1h interval):
- 12 hours → 12 bars (intraday trends)
- 24 hours → 24 bars (daily cycle)
- 48 hours → 48 bars (2-day swing)
- 168 hours → 168 bars (weekly trend)

**Effect on strategy**:
- **Lower values**: More responsive, captures short-term trends, more trades
- **Higher values**: Smoother, captures long-term trends, fewer trades

**Recommendations by interval**:
| Interval | Min Period (hours) | Bars | Strategy Type |
|----------|-------------------|------|---------------|
| 30m | 12-24 | 24-48 | Scalping/day trading |
| 1h | 24-48 | 24-48 | Day trading |
| 4h | 48-168 | 12-42 | Swing trading |

### Cutoff Scale

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `cutoff_scale` | float | 0.3-3.0 | `1.0` | Multiplier for cutoff frequency |

**Effect**:
```python
actual_cutoff = (1.0 / min_period_bars) * cutoff_scale
```

- **< 1.0**: More smoothing (lower cutoff frequency)
  - 0.5: Half the cutoff, preserves 2x longer trends
  - Effect: Smoother line, fewer noise-induced signals
  
- **= 1.0**: Exact cutoff at min trend period
  
- **> 1.0**: Less smoothing (higher cutoff frequency)
  - 2.0: Double the cutoff, allows shorter trends
  - Effect: More responsive to price changes, more signals

**Tuning guide**:
| Market Condition | Recommended Scale |
|-----------------|------------------|
| High volatility (crypto 2021) | 0.5-0.7 |
| Normal volatility | 0.8-1.2 |
| Low volatility (stablecoin pairs) | 1.5-2.5 |

### Spectral Analysis Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `window_length` | int | 64-512 | `256` | Window size for Welch PSD (bars) |
| `overlap_ratio` | float | 0.0-0.75 | `0.5` | Overlap between windows |

**Window length**:
- **Too small** (< 64): Poor frequency resolution, noisy spectrum
- **Optimal** (128-256): Good balance for most use cases
- **Too large** (> 512): Few windows, high variance in estimates

**Overlap ratio**:
- **0%**: No overlap, maximum independence
- **50%**: Standard, good variance reduction
- **75%**: High overlap, smoothest estimates but slower

**Performance impact**:
| Window Length | Overlap | Computation Time |
|--------------|---------|-----------------|
| 64 | 50% | Fast (< 1s) |
| 256 | 50% | Moderate (~2s) |
| 512 | 75% | Slow (~5s) |

---

## Signal Generation Parameters

### Slope Threshold

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `slope_threshold` | float | -0.01 to 0.01 | `0.0` | Minimum trend slope for entry signal |

**Units**: Slope in price/bar (e.g., $100/bar for BTC at $40k would be ~0.0025 or 0.25%)

**Calculation**:
```python
slope[t] = (smoothed[t] - smoothed[t - lookback]) / lookback
```

**Effect**:
- **0.0**: Enter on any uptrend (price above smoothed)
- **> 0.0**: Require minimum momentum
  - 0.0001: Very weak filter
  - 0.001: Moderate filter (recommended)
  - 0.01: Strong filter (only steep trends)

**Recommendations**:
| Asset Volatility | Threshold Range |
|-----------------|----------------|
| Low (BTC/ETH) | 0.0-0.0005 |
| Medium (Top 20 alts) | 0.0005-0.002 |
| High (Small caps) | 0.002-0.01 |

### Slope Lookback

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `slope_lookback` | int | 1-10 | `1` | Number of bars to compute slope |

**Effect**:
- **1 bar**: Instant slope (noisy)
- **3-5 bars**: Smoothed slope (recommended)
- **> 5 bars**: Very smooth but laggy

**Trade-off**:
- Lower: More responsive, more signals, more false positives
- Higher: Smoother, fewer signals, fewer false positives

---

## Stop Loss Parameters

### Stop Type

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `stop_type` | string | ATR, Residual | Method for computing stop levels |

**ATR (Average True Range)**:
- Based on price volatility
- Adapts to market conditions
- Best for: Momentum/breakout strategies

**Residual**:
- Based on price-trend deviation
- Tighter in stable trends
- Best for: Pure trend-following

### ATR Period

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `atr_period` | int | 7-30 | `14` | Lookback period for ATR calculation |

**Effect**:
- **7-10**: Short-term volatility (responsive)
- **14**: Standard (balanced)
- **20-30**: Long-term volatility (stable)

**Recommendations by interval**:
| Interval | ATR Period |
|----------|-----------|
| 30m | 10-14 |
| 1h | 14-20 |
| 4h | 14-21 |

### Residual Window

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `residual_window` | int | 10-50 | `20` | Lookback for residual standard deviation |

**Used only for Residual stop type**

**Effect**:
- **10-15**: Responsive to recent deviations
- **20-30**: Balanced
- **> 30**: Stable, based on long-term deviation

### K Stop (Stop Loss Multiplier)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `k_stop` | float | 1.0-4.0 | `2.0` | Multiplier for stop loss distance |

**Formula**:
```python
# ATR-based
stop_loss = entry_price - k_stop * ATR

# Residual-based
stop_loss = smoothed - k_stop * sigma_residual
```

**Effect**:
- **1.0-1.5**: Very tight stops
  - Pros: Limits losses
  - Cons: Frequent whipsaws, low win rate
  
- **2.0-2.5**: Moderate stops (recommended)
  - Balanced risk/reward
  
- **3.0-4.0**: Wide stops
  - Pros: Avoids noise
  - Cons: Large losses when wrong

**Optimization approach**:
1. Start with 2.0
2. If win rate < 40%, increase (too tight)
3. If avg loss > 2x avg win, decrease (too wide)

### K Profit (Take Profit Multiplier)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `k_profit` | float | 2.0-6.0 | `3.0` | Multiplier for take profit distance |

**Formula**:
```python
take_profit = entry_price + k_profit * ATR
```

**Relationship with K Stop**:
```python
risk_reward_ratio = k_profit / k_stop
```

**Recommendations**:
| K Stop | K Profit | R:R Ratio |
|--------|----------|-----------|
| 1.5 | 3.0 | 2:1 |
| 2.0 | 3.0 | 1.5:1 |
| 2.0 | 4.0 | 2:1 |
| 2.5 | 5.0 | 2:1 |

**Rule of thumb**: `k_profit >= 1.5 * k_stop` for positive expectancy

---

## Backtest Configuration

### Initial Capital

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `initial_capital` | float | 100-1,000,000+ | `10000.0` | Starting portfolio value ($) |

**Effect**:
- Scales absolute returns
- Does not affect percentage returns or Sharpe ratio
- Affects position sizing (if using fixed $ amounts)

**Typical values**:
- **$10,000**: Personal account testing
- **$100,000**: Small fund
- **$1,000,000**: Large fund

### Fee Rate

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `fee_rate` | float | 0.0-0.01 | `0.001` | Trading fee as decimal (0.1% = 0.001) |

**Round-trip cost**:
```python
total_cost = fee_rate * 2  # Entry + exit
```

**Platform fees**:
| Exchange | Spot Fee | Futures Fee |
|----------|----------|-------------|
| Binance | 0.1% (0.001) | 0.04% maker, 0.0675% taker |
| Coinbase | 0.5% (0.005) | 0.4% |
| Kraken | 0.16-0.26% | 0.02-0.05% |

**Impact on strategy**:
- High-frequency (many trades): Fees dominate, need tight spread
- Low-frequency (few trades): Fees minor, can tolerate higher

### Slippage

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `slippage` | float | 0.0-0.01 | `0.0005` | Expected slippage per trade (0.05% = 0.0005) |

**Modeling approaches**:
1. **Fixed %**: Same slippage regardless of size (default)
2. **Volume-based**: `slippage = f(order_size / avg_volume)`
3. **Spread-based**: `slippage = bid_ask_spread / 2`

**Typical values**:
| Market Condition | Slippage |
|-----------------|----------|
| High liquidity (BTC, ETH) | 0.01-0.05% |
| Medium liquidity (top 20) | 0.05-0.1% |
| Low liquidity (small caps) | 0.1-0.5% |

**Total transaction cost**:
```python
total_cost_per_trade = fee_rate + slippage
# Example: 0.001 + 0.0005 = 0.0015 (0.15% per side)
# Round-trip: 0.3%
```

---

## Position Sizing Parameters

**Location**: M7 enhanced backtester

### Sizing Mode

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `sizing_mode` | string | full, fixed, volatility, risk | Position sizing method |

**Modes**:
1. **full**: Use 100% of capital per trade
2. **fixed**: Fixed dollar amount or percentage
3. **volatility**: Scale by volatility target
4. **risk**: Scale by risk per trade

### Position Size Fraction

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `position_size_fraction` | float | 0.1-1.0 | `1.0` | Fraction of capital per trade (for 'fixed' mode) |

**Examples**:
- `1.0`: Full capital (100%)
- `0.5`: Half capital (50%)
- `0.2`: One-fifth capital (20%)

**Use case**: Risk management, correlation hedging

### Volatility Target

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `volatility_target` | float | 0.01-0.05 | `0.02` | Target daily portfolio volatility (for 'volatility' mode) |

**Formula**:
```python
position_size = capital * (volatility_target / asset_volatility)
```

**Example**:
- Capital: $10,000
- Target vol: 2% = $200 daily moves
- BTC volatility: 4%
- Position: $10,000 * (0.02 / 0.04) = $5,000 (50%)

### Allow Pyramiding

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `allow_pyramiding` | bool | `False` | Allow adding to existing positions |

**If enabled**:
- Can enter multiple times in same direction
- Max positions controlled by `max_pyramid_levels`

**Use case**: Trend continuation, dollar-cost averaging into trends

---

## Multi-Timeframe Parameters

**Location**: M7 multi-timeframe module

### Use MTF Confirmation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_mtf_confirmation` | bool | `False` | Require higher timeframe trend agreement |

### Higher Timeframes

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `confirm_tf_1` | string | 1h, 4h | First confirmation timeframe |
| `confirm_tf_2` | string | 4h, 1d | Second confirmation timeframe |

**Rules**:
- Confirmation TF must be > execution TF
- Example: Execute on 30m, confirm with 1h and 4h

### MTF Smoothing Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `confirm_tf_1_period` | float | Min trend period for TF1 (hours) |
| `confirm_tf_2_period` | float | Min trend period for TF2 (hours) |

**Recommendations**:
| Execution TF | Confirm TF1 | TF1 Period | Confirm TF2 | TF2 Period |
|-------------|-------------|-----------|-------------|-----------|
| 30m | 1h | 6-12h | 4h | 24-48h |
| 1h | 4h | 12-24h | 1d | 48-96h |

---

## Optimization Parameters

**Location**: M8 optimization module

### Optimization Method

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `optimization_method` | string | grid, random, bayesian | Search algorithm |

### Grid Search Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `param_ranges` | dict | - | - | Dictionary of parameter names to lists of values |

**Example**:
```python
param_ranges = {
    'min_trend_hours': [12, 24, 48],
    'cutoff_scale': [0.8, 1.0, 1.2],
    'k_stop': [1.5, 2.0, 2.5]
}
# Total combinations: 3 * 3 * 3 = 27
```

### Random Search Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `n_iter` | int | 10-1000 | `100` | Number of random samples |

### Bayesian Optimization Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `n_initial` | int | 5-20 | `10` | Initial random samples |
| `n_iter` | int | 20-200 | `50` | Optimization iterations |
| `acquisition_func` | string | EI, LCB, PI | `EI` | Acquisition function |

**Acquisition functions**:
- **EI** (Expected Improvement): Balanced exploration/exploitation
- **LCB** (Lower Confidence Bound): More exploration
- **PI** (Probability of Improvement): More exploitation

### Walk-Forward Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `train_window_days` | int | 90-730 | `180` | Training period length |
| `test_window_days` | int | 30-180 | `60` | Testing period length |
| `mode` | string | rolling, anchored | `rolling` | Window advancement mode |

**Mode comparison**:
- **Rolling**: Fixed-size train window slides forward
- **Anchored**: Train window grows, always starts at beginning

### Monte Carlo Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `n_simulations` | int | 100-10000 | `1000` | Number of bootstrap simulations |
| `block_size` | int | 5-50 | `20` | Block size for bootstrap |

---

## Portfolio Parameters

**Location**: M9 portfolio module

### Symbol Selection

| Parameter | Type | Description |
|-----------|------|-------------|
| `symbols` | list[str] | List of symbols to include in portfolio |

**Example**:
```python
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT']
```

### Weighting Method

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `weighting_method` | string | equal, volatility, risk_parity, market_cap | Asset allocation method |

**Characteristics**:
| Method | Diversification | Computation | Best For |
|--------|----------------|-------------|---------|
| Equal | High | Fast | Baseline, many assets |
| Volatility | Medium | Fast | Risk normalization |
| Risk Parity | Highest | Slow (optimization) | Maximum diversification |
| Market Cap | Low | Fast | Index tracking |

### Rebalancing Frequency

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `rebalance_frequency` | string | daily, weekly, monthly, threshold | How often to rebalance |

**Trade-off**:
- **High frequency** (daily): Maintains target weights, high transaction costs
- **Low frequency** (monthly): Lower costs, weight drift
- **Threshold-based**: Best of both (rebalance only when needed)

### Rebalance Threshold

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `rebalance_threshold` | float | 0.01-0.2 | `0.05` | Weight drift threshold for rebalancing |

**Example**:
```python
# Target weight: 25%
# Current weight: 30%
# Drift: 5% (0.05)
# If threshold = 0.05, rebalance triggered
```

### Portfolio-Level Constraints

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `max_position_size` | float | 0.1-1.0 | `0.5` | Maximum weight per asset |
| `max_leverage` | float | 1.0-5.0 | `1.0` | Maximum portfolio leverage |

---

## Presets and Defaults

### Default Configuration

**File**: `config/settings.py`

```python
class UIConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    start_date: datetime = datetime(2024, 1, 1, tzinfo=UTC)
    end_date: datetime = datetime(2024, 6, 1, tzinfo=UTC)
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
    
    initial_capital: float = 10000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
```

### Preset Examples

#### Conservative Trend Following
```yaml
name: "Conservative Trend"
min_trend_hours: 48
cutoff_scale: 0.8
k_stop: 2.5
k_profit: 4.0
slope_threshold: 0.001
```

#### Aggressive Scalping
```yaml
name: "Aggressive Scalp"
interval: "30m"
min_trend_hours: 12
cutoff_scale: 1.5
k_stop: 1.5
k_profit: 2.5
slope_threshold: 0.0
```

#### Volatility-Adaptive
```yaml
name: "Vol Adaptive"
stop_type: "ATR"
atr_period: 10
k_stop: 2.0
sizing_mode: "volatility"
volatility_target: 0.02
```

### Saving Custom Presets

1. Configure parameters in UI
2. Navigate to "💾 Presets & Persistence"
3. Enter preset name
4. Click "Save Current Config"

**File location**: Defined by `PRESET_STORAGE_PATH` env variable

**Format**: YAML
```yaml
presets:
  MyPreset:
    symbol: BTCUSDT
    interval: 1h
    min_trend_hours: 24.0
    # ... all parameters
```

---

## Parameter Interaction Matrix

Some parameters interact in non-obvious ways:

| Parameter 1 | Parameter 2 | Interaction |
|------------|-------------|-------------|
| `min_trend_hours` | `cutoff_scale` | Combined determine smoothing aggressiveness |
| `k_stop` | `k_profit` | Define risk/reward ratio |
| `slope_threshold` | `min_trend_hours` | Longer trends need lower slope thresholds |
| `fee_rate` | `slippage` | Combined determine transaction cost |
| `sizing_mode` | `volatility_target` | Volatility target only used in 'volatility' mode |
| `interval` | `min_trend_hours` | Must convert hours to bars for smoothing |

---

## Optimization Tips

### For Beginners

Start with these parameters:
1. `min_trend_hours`: 24-48
2. `cutoff_scale`: 0.8-1.2
3. `k_stop`: 1.5-2.5
4. `k_profit`: 2.5-4.0

Fix everything else at defaults.

### For Intermediate Users

Optimize in stages:
1. **Stage 1**: Smoothing (`min_trend_hours`, `cutoff_scale`)
2. **Stage 2**: Stops (`k_stop`, `k_profit`)
3. **Stage 3**: Signals (`slope_threshold`, `slope_lookback`)

### For Advanced Users

Full optimization with walk-forward:
1. Define wide parameter ranges
2. Run Bayesian optimization (50-100 iterations)
3. Take top 5 configs
4. Run walk-forward validation
5. Choose most robust (least IS/OOS degradation)

---

## Validation Checklist

Before running optimization, ensure:

- [ ] `start_date` < `end_date`
- [ ] Date range sufficient (> 3 months for testing)
- [ ] `k_profit` > `k_stop` (positive risk/reward)
- [ ] `slope_threshold` appropriate for asset volatility
- [ ] `fee_rate` + `slippage` < 0.01 (< 1% total cost)
- [ ] `min_trend_hours` > `interval` duration
- [ ] For MTF: confirmation TFs > execution TF

---

## Further Reading

- [ARCHITECTURE.md](ARCHITECTURE.md) - Implementation details
- [FAQ.md](FAQ.md) - Common configuration questions
- [README.md](../README.md) - Usage guide
