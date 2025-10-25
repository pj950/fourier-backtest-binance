# Binance Fourier Backtester

A Python 3.11 + Streamlit application for fetching, caching, and visualizing Binance OHLCV data with advanced Fourier analysis, spectral smoothing, and algorithmic trading backtesting capabilities.

## Features

### Data & Analysis
- **Data Fetching**: Fetch 30m, 1h, and 4h klines from Binance REST API with automatic retries and rate limiting
- **Smart Caching**: Parquet-based caching with incremental updates and automatic gap detection/backfill
- **DCT Smoothing**: Discrete Cosine Transform-based low-pass smoothing with mirrored padding and tapered cutoff
- **FFT Spectrum Analysis**: Global power spectrum with dominant frequency peaks labeled in bars/hours
- **Sliding Window PSD**: Welch's method for local dominant period extraction over time
- **Spectral Heatmaps**: Time-frequency analysis showing how dominant periods evolve

### Backtesting & Trading
- **Dynamic Stop Bands**: ATR-based and residual-based stops with configurable multipliers
- **Signal Generation**: Trend-following signals with slope and volatility filters
- **Multi-Timeframe Confirmation**: Execute on 30m with 1h/4h trend filters for higher probability setups
- **Advanced Exits**: Time-based stops, partial take-profit scaling, slope reversal confirmation
- **Dynamic Position Sizing**: Volatility-based (ATR/sigma), fixed risk, optional pyramiding
- **Short/Futures Trading**: Optional short trading mode with configurable fees per venue
- **Vectorized Backtester**: Fast, realistic backtesting with next-bar fills, fees, and slippage
- **Performance Metrics**: 19 metrics including Sharpe, Sortino, win rate, profit factor, and more
- **Trade Analysis**: MAE/MFE tracking, equity curve, complete trade logs with exit reasons

### Parameter Optimization & Robustness (M8)
- **Grid/Random/Bayesian Search**: Multiple optimization algorithms for parameter tuning
- **Walk-Forward Analysis**: Rolling or anchored validation with train/test splits
- **Monte Carlo Resampling**: Block bootstrap for robustness evaluation
- **Rich Visualizations**: Heatmaps, frontier plots, parameter importance, progress tracking
- **Export Capabilities**: CSV/Parquet export with best configurations
- **Reproducible Seeds**: All methods support seeding for reproducibility
- **Batch Processing**: Leaderboard-based evaluation of parameter combinations

### Portfolio & Risk Management (M9)
- **Multi-Symbol Backtesting**: Parallel per-symbol runs with portfolio aggregation
- **Weighting Schemes**: Equal, volatility-scaled, risk parity, market cap weighted
- **Dynamic Rebalancing**: Configurable frequency and threshold-based rebalancing
- **Correlation Analysis**: Static and rolling correlation matrices
- **Risk Analytics**: Diversification ratio, concentration metrics, risk contributions
- **Exposure Tracking**: Sector exposure and beta calculations
- **Portfolio Metrics**: Comprehensive portfolio-level performance and risk metrics
- **Interactive Portfolio UI**: Symbol basket selection, weight visualization, equity curves

### Infrastructure
- **Interactive UI**: Streamlit-based interface with real-time parameter adjustment
- **Multiple Modes**: Backtesting, Optimization, and Portfolio tabs
- **Robust Error Handling**: Exponential backoff retries with configurable limits
- **Type Safety**: Full type hints with mypy validation
- **Comprehensive Testing**: 150+ tests covering all components including M7, M8, and M9

## Installation

### Using Poetry (Recommended)

```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Run the application
streamlit run app/ui/main.py
```

### Using Docker

```bash
# Build the image
docker build -t binance-backtester .

# Run the container
docker run -p 8501:8501 -v $(pwd)/data:/app/data binance-backtester
```

## Configuration

Copy `.env.example` to `.env` and customize as needed:

```bash
cp .env.example .env
```

Key configuration options:
- `BINANCE_BASE_URL`: Binance API endpoint
- `MAX_RETRY_ATTEMPTS`: Number of retry attempts for failed requests
- `DEFAULT_FEE_RATE`: Default trading fee rate
- `CACHE_DIR`: Directory for storing cached Parquet files
- `PRESET_STORAGE_PATH`: Path to the YAML file that stores named parameter presets
- `LAST_SESSION_STATE_PATH`: Path to the YAML file that persists the most recent UI state

## Project Structure

```
.
├── app/
│   └── ui/
│       ├── main.py                 # Streamlit UI entry point
│       ├── optimization_tab.py     # Optimization interface (M8)
│       └── portfolio_tab.py        # Portfolio interface (M9)
├── core/
│   ├── analysis/
│   │   ├── fourier.py              # DCT-based smoothing functions
│   │   ├── spectral.py             # FFT/Welch PSD analysis
│   │   ├── signals.py              # Signal generation
│   │   ├── stops.py                # Stop loss/take profit
│   │   ├── mtf.py                  # Multi-timeframe analysis (M7)
│   │   ├── exits.py                # Additional exit strategies (M7)
│   │   └── sizing.py               # Position sizing (M7)
│   ├── backtest/
│   │   └── engine.py               # Backtesting engine with M7 enhancements
│   ├── data/
│   │   ├── binance_client.py       # Binance API client
│   │   ├── cache.py                # Parquet caching with gap detection
│   │   └── loader.py               # Unified data loading API
│   ├── optimization/               # M8 optimization framework
│   │   ├── search.py               # Grid/Random/Bayesian search
│   │   ├── params.py               # Parameter space definitions
│   │   ├── walkforward.py          # Walk-forward analysis
│   │   ├── monte_carlo.py          # Monte Carlo resampling
│   │   └── visualization.py        # Optimization visualizations
│   ├── portfolio/                  # M9 portfolio management
│   │   ├── portfolio.py            # Main portfolio manager
│   │   ├── weights.py              # Weighting schemes
│   │   ├── analytics.py            # Risk and correlation analytics
│   │   └── executor.py             # Parallel backtest execution
│   └── utils/
│       └── time.py                 # UTC time utilities
├── config/
│   └── settings.py                 # Configuration management
├── examples/
│   ├── mtf_strategy_example.py    # Complete MTF strategy example (M7)
│   └── portfolio_example.py        # Portfolio management example (M9)
├── tests/
│   ├── test_backtest.py            # Backtest engine tests
│   ├── test_backtest_enhanced.py   # Enhanced backtest tests (M7)
│   ├── test_data_fetch_cache.py    # Data layer tests
│   ├── test_fourier.py             # Fourier smoothing tests
│   ├── test_spectral.py            # Spectral analysis tests
│   ├── test_mtf.py                 # Multi-timeframe tests (M7)
│   ├── test_exits.py               # Exit strategies tests (M7)
│   ├── test_sizing.py              # Position sizing tests (M7)
│   ├── test_optimization.py        # Optimization tests (M8)
│   ├── test_portfolio.py           # Portfolio tests (M9)
│   ├── test_portfolio_weights.py   # Weighting scheme tests (M9)
│   ├── test_portfolio_analytics.py # Analytics tests (M9)
│   └── test_strategy_integration.py # Integration tests (M7)
├── IMPLEMENTATION_M7.md            # M7 features documentation
├── IMPLEMENTATION_M8.md            # M8 features documentation
├── IMPLEMENTATION_M9.md            # M9 features documentation
├── Dockerfile                      # Docker configuration
├── pyproject.toml                  # Poetry dependencies
└── README.md                       # This file
```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app/ui/main.py
   ```

2. Open your browser to `http://localhost:8501`

3. Select your parameters:
   - **Symbol**: BTCUSDT or ETHUSDT
   - **Interval**: 30m, 1h, or 4h
   - **Date Range**: Start and end dates for data

4. Click "Load Data" to fetch and visualize the OHLCV data

5. Configure Fourier analysis parameters:
   - **Min Trend Period**: Minimum period to preserve (in hours)
   - **Cutoff Scale**: Smoothing aggressiveness (higher = more smoothing)
   - **Window Length**: Bars per window for Welch PSD (64-512)
   - **Window Overlap**: Overlap between consecutive windows (0-75%)

6. Enable visualizations:
   - **DCT Smoothing**: Shows smoothed price overlaid on candlesticks
   - **FFT Spectrum**: Global frequency analysis with dominant peaks
   - **Sliding Window Dominant Period**: Time-varying dominant cycles
   - **Welch PSD Heatmap**: Time-frequency spectral density map

## Data Caching

The application uses Parquet files to cache OHLCV data locally:
- Files are stored in `data/cache/` with naming pattern `{SYMBOL}_{INTERVAL}.parquet`
- Incremental updates fetch only new data since the last cached bar
- Gap detection automatically identifies and backfills missing data
- Data is deduplicated and sorted by timestamp

## Parameter Presets & Session Persistence

Open the **💾 Presets & Persistence** expander in the Streamlit UI to:
- Save the current configuration under a custom name
- Load or delete previously saved presets
- View which preset (or last session) is currently active

The app automatically restores the last parameters you used on startup and keeps
that state in sync as you adjust controls. Presets and the last session snapshot
are stored as YAML files whose locations are configurable via the
`PRESET_STORAGE_PATH` and `LAST_SESSION_STATE_PATH` environment variables.

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=config --cov=app

# Run specific test file
pytest tests/test_data_fetch_cache.py
```

## Development

The project uses:
- **Ruff** for linting and formatting
- **mypy** for type checking
- **pytest** for testing

Run checks:

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type check
mypy .

# Run tests
pytest
```

## API Reference

### Data Loading

#### `load_klines(symbol, interval, start, end, force_refresh=False)`

Load OHLCV data for a given symbol and time range.

**Parameters:**
- `symbol` (str): Trading pair (e.g., "BTCUSDT")
- `interval` (str): Timeframe ("30m", "1h", "4h")
- `start` (datetime): Start time (UTC)
- `end` (datetime): End time (UTC)
- `force_refresh` (bool): Bypass cache and fetch fresh data

**Returns:**
- `pd.DataFrame`: OHLCV data with columns: open_time, open, high, low, close, volume, quote_volume, trades, close_time

### Fourier Analysis

#### `dct_lowpass_smooth(signal, cutoff_freq, taper_width=0.1, padding_ratio=0.2)`

Apply DCT-based low-pass smoothing with mirrored padding.

**Parameters:**
- `signal` (np.ndarray): 1D input signal
- `cutoff_freq` (float): Cutoff frequency as fraction of Nyquist (0.0 to 1.0)
- `taper_width` (float): Width of taper region as fraction of cutoff
- `padding_ratio` (float): Ratio of signal length for mirrored padding

**Returns:**
- `np.ndarray`: Smoothed signal (same length as input)

#### `smooth_price_series(prices, min_period_bars, cutoff_scale=1.0, taper_width=0.1)`

Smooth a price series using DCT low-pass filter.

**Parameters:**
- `prices` (np.ndarray): Price array
- `min_period_bars` (int): Minimum trend period to preserve (in bars)
- `cutoff_scale` (float): Scale factor for cutoff frequency
- `taper_width` (float): Width of taper region

**Returns:**
- `np.ndarray`: Smoothed price series

### Spectral Analysis

#### `compute_fft_spectrum(signal, sampling_rate=1.0)`

Compute FFT power spectrum of a signal.

**Parameters:**
- `signal` (np.ndarray): Input signal
- `sampling_rate` (float): Sampling rate (bars per unit time)

**Returns:**
- `tuple[np.ndarray, np.ndarray]`: Frequencies and power spectrum

#### `compute_welch_psd(signal, window_length=256, overlap_ratio=0.5, sampling_rate=1.0)`

Compute power spectral density using Welch's method.

**Parameters:**
- `signal` (np.ndarray): Input signal
- `window_length` (int): Length of each window segment
- `overlap_ratio` (float): Overlap between segments (0.0 to 1.0)
- `sampling_rate` (float): Sampling rate

**Returns:**
- `tuple[np.ndarray, np.ndarray]`: Frequencies and power spectral density

#### `compute_sliding_dominant_period(signal, window_length=256, overlap_ratio=0.5, sampling_rate=1.0)`

Compute dominant period over time using sliding windows.

**Parameters:**
- `signal` (np.ndarray): Input signal
- `window_length` (int): Window length for each PSD computation
- `overlap_ratio` (float): Overlap between windows
- `sampling_rate` (float): Sampling rate

**Returns:**
- `tuple[np.ndarray, np.ndarray]`: Time indices and dominant periods

### Dynamic Stop Bands

#### `compute_atr(high, low, close, period=14)`

Compute Average True Range (ATR).

**Parameters:**
- `high` (np.ndarray): High prices
- `low` (np.ndarray): Low prices
- `close` (np.ndarray): Close prices
- `period` (int): ATR period

**Returns:**
- `np.ndarray`: ATR values

#### `compute_atr_stops(close, high, low, atr_period=14, k_stop=2.0, k_profit=3.0)`

Compute ATR-based stop and take-profit bands.

**Parameters:**
- `close` (np.ndarray): Close prices
- `high` (np.ndarray): High prices
- `low` (np.ndarray): Low prices
- `atr_period` (int): Period for ATR calculation
- `k_stop` (float): Multiplier for stop loss
- `k_profit` (float): Multiplier for take profit

**Returns:**
- `tuple[np.ndarray, ...]`: (long_stop, long_profit, short_stop, short_profit)

#### `compute_residual_stops(close, smoothed, method='sigma', window=20, quantile=0.95, k_stop=2.0, k_profit=3.0)`

Compute residual-based stop and take-profit bands.

**Parameters:**
- `close` (np.ndarray): Close prices
- `smoothed` (np.ndarray): Smoothed trend line
- `method` (str): 'sigma' or 'quantile'
- `window` (int): Rolling window size
- `quantile` (float): Quantile level (if method='quantile')
- `k_stop` (float): Multiplier for stop loss
- `k_profit` (float): Multiplier for take profit

**Returns:**
- `tuple[np.ndarray, ...]`: (long_stop, long_profit, short_stop, short_profit)

### Signal Generation

#### `generate_trend_following_signals(close, smoothed, slope_threshold=0.0, slope_lookback=1, min_volatility=0.0, volatility=None)`

Generate trend-following entry and exit signals.

**Parameters:**
- `close` (np.ndarray): Close prices
- `smoothed` (np.ndarray): Smoothed trend line
- `slope_threshold` (float): Minimum slope for entry
- `slope_lookback` (int): Lookback for slope computation
- `min_volatility` (float): Minimum volatility filter
- `volatility` (np.ndarray): Volatility measure (optional)

**Returns:**
- `tuple[np.ndarray, np.ndarray]`: (entry_signals, exit_signals) as boolean arrays

#### `generate_signals_with_stops(close, smoothed, stop_levels, slope_threshold=0.0, slope_lookback=1, min_volatility=0.0, volatility=None)`

Generate signals with integrated stop loss logic.

**Parameters:**
- Same as `generate_trend_following_signals` plus:
- `stop_levels` (np.ndarray): Stop loss levels

**Returns:**
- `np.ndarray`: Signal array (1=entry, -1=exit, 0=hold)

### Backtesting

#### `run_backtest(signals, open_prices, high_prices, low_prices, close_prices, timestamps, config=None)`

Run vectorized backtest with next-bar open fills.

**Parameters:**
- `signals` (np.ndarray): Signal array (1=entry, -1=exit, 0=hold)
- `open_prices` (np.ndarray): Open prices
- `high_prices` (np.ndarray): High prices
- `low_prices` (np.ndarray): Low prices
- `close_prices` (np.ndarray): Close prices
- `timestamps` (pd.DatetimeIndex): Timestamps
- `config` (BacktestConfig): Configuration (optional)

**Returns:**
- `BacktestResult`: Results with equity_curve, trades, and metrics

#### `BacktestConfig` Parameters

- `initial_capital` (float): Starting capital (default 10,000)
- `fee_rate` (float): Trading fee rate (default 0.001)
- `slippage` (float): Slippage per trade (default 0.0005)
- `position_size_mode` (str): 'full' or 'fixed'
- `position_size_fraction` (float): Fraction of capital (default 1.0)

#### Performance Metrics

The `BacktestResult.metrics` dictionary contains:
- `total_return`, `cumulative_return`, `annualized_return`
- `max_drawdown`, `max_drawdown_pct`
- `sharpe_ratio`, `sortino_ratio`
- `n_trades`, `n_wins`, `n_losses`, `win_rate`
- `profit_factor`, `avg_win`, `avg_loss`
- `avg_bars_held`, `avg_mae`, `avg_mfe`, `avg_mae_pct`, `avg_mfe_pct`

## Troubleshooting

### Binance Rate Limits

If the UI reports a Binance rate limit error:
- Wait for the suggested retry window before reloading data
- Narrow the requested date range to reduce API calls
- Leave caching enabled (disable **Force Refresh**) so repeated requests reuse local data
- Avoid running multiple backtests with force refresh simultaneously

## Example Usage

### Complete Backtest Workflow

```python
from core.data.loader import load_klines
from core.analysis.fourier import smooth_price_series
from core.analysis.stops import compute_atr_stops
from core.analysis.signals import generate_signals_with_stops
from core.backtest.engine import BacktestConfig, run_backtest

# Load data
df = load_klines("BTCUSDT", "1h", start_date, end_date)

# Extract OHLCV
close = df["close"].values
high = df["high"].values
low = df["low"].values
open_prices = df["open"].values
timestamps = df["open_time"]

# Smooth prices
smoothed = smooth_price_series(close, min_period_bars=24, cutoff_scale=1.0)

# Compute stop bands
long_stop, long_profit, _, _ = compute_atr_stops(
    close, high, low, atr_period=14, k_stop=2.0, k_profit=3.0
)

# Generate signals
signals = generate_signals_with_stops(
    close=close,
    smoothed=smoothed,
    stop_levels=long_stop,
    slope_threshold=0.0
)

# Run backtest
config = BacktestConfig(
    initial_capital=10000,
    fee_rate=0.001,
    slippage=0.0005
)
result = run_backtest(signals, open_prices, high, low, close, timestamps, config)

# Print results
print(f"Total return: {result.metrics['total_return']:.2%}")
print(f"Sharpe ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Win rate: {result.metrics['win_rate']:.2%}")
print(f"Number of trades: {result.metrics['n_trades']}")
```

## M7 Strategy Enhancements

The M7 milestone adds advanced trading strategy features. See [IMPLEMENTATION_M7.md](IMPLEMENTATION_M7.md) for detailed documentation.

### Multi-Timeframe Strategy Example

```python
from core.analysis.mtf import align_timeframes, compute_trend_direction, check_mtf_alignment
from core.backtest.engine import BacktestConfig, run_backtest_enhanced

# Load multiple timeframes
df_30m = load_klines("BTCUSDT", "30m", start_date, end_date)
df_1h = load_klines("BTCUSDT", "1h", start_date, end_date)
df_4h = load_klines("BTCUSDT", "4h", start_date, end_date)

# Align and compute trends
df_aligned = align_timeframes(df_30m, df_1h, "30m", "1h")
trend_30m = compute_trend_direction(close, smoothed_30m)
trend_1h = compute_trend_direction(close, smoothed_1h)
trend_4h = compute_trend_direction(close, smoothed_4h)

# Filter by alignment
aligned_long, aligned_short = check_mtf_alignment(trend_30m, trend_1h, trend_4h)

# Run enhanced backtest with dynamic sizing
config = BacktestConfig(
    initial_capital=10000.0,
    allow_shorts=False,
    max_bars_held=100,
    sizing_mode="volatility",
    volatility_target=0.02,
)

result = run_backtest_enhanced(signals, open_prices, high, low, close, timestamps, 
                               atr=atr, stop_levels=stops, config=config)
```

For a complete working example, see `examples/mtf_strategy_example.py`.

## M8 Parameter Optimization

The M8 milestone adds parameter optimization and robustness evaluation. See [IMPLEMENTATION_M8.md](IMPLEMENTATION_M8.md) for detailed documentation.

### Optimization Example

```python
from core.optimization.params import create_default_param_space
from core.optimization.runner import OptimizationRunner

# Define objective function
def objective_function(params, df):
    result = run_strategy_with_params(params, df)
    return result.metrics

# Create runner
runner = OptimizationRunner(
    objective_function=objective_function,
    objective_metric="sharpe_ratio",
    maximize=True,
    seed=42,
)

# Run Bayesian optimization
opt_run = runner.run_bayesian_search(
    param_spaces=create_default_param_space(),
    data=df,
    n_initial=10,
    n_iter=40,
)

print(f"Best Sharpe: {opt_run.best_score:.4f}")
print(f"Best params: {opt_run.best_params}")

# Export results
export_full_optimization_results(opt_run, "results/", include_visualizations=True)
```

For a complete working example with walk-forward and Monte Carlo, see `examples/optimization_example.py`.

## M9 Portfolio & Risk Management

The M9 milestone adds multi-symbol portfolio management and risk controls. See [IMPLEMENTATION_M9.md](IMPLEMENTATION_M9.md) for detailed documentation.

### Portfolio Example

```python
from datetime import UTC, datetime
from core.data.loader import load_klines
from core.portfolio.portfolio import create_portfolio

# Load data for multiple symbols
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

# Create portfolio with risk parity weights
portfolio = create_portfolio(
    symbols=symbols,
    weighting_method="risk_parity",
    initial_capital=10000.0,
)

# Run portfolio backtest
result = portfolio.run_backtest(
    data_dict=data_dict,
    strategy_func=my_strategy,
    strategy_params={},
)

# Analyze results
print(f"Total Return: {result.metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Diversification: {result.metrics['diversification_ratio']:.2f}")
print(f"Correlation Matrix:\n{result.correlation_matrix}")

# Compare weights
for symbol, weight in zip(symbols, result.weights):
    print(f"{symbol}: {weight:.2%}")
```

Available weighting methods:
- **Equal**: Simple equal weighting
- **Volatility**: Inverse volatility weighting
- **Risk Parity**: Equal risk contribution
- **Market Cap**: Market capitalization weighted

For a complete working example comparing different weighting schemes, see `examples/portfolio_example.py`.

## License

MIT License
