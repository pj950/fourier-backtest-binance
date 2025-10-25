# Binance Fourier Backtester

A comprehensive Python 3.11+ trading strategy platform combining advanced Fourier analysis, spectral smoothing, and algorithmic backtesting with an interactive Streamlit UI. Designed for systematic trading research, parameter optimization, and portfolio risk management.

## 🎯 Project Overview

This platform enables quantitative traders and researchers to:
- **Analyze** cryptocurrency price data using Fourier transforms and spectral methods
- **Develop** trend-following strategies with dynamic stops and multi-timeframe confirmation
- **Backtest** strategies with realistic fills, fees, and slippage modeling
- **Optimize** parameters using grid search, random search, or Bayesian optimization
- **Manage** multi-symbol portfolios with advanced risk analytics

### Key Use Cases
- Identifying dominant market cycles using FFT and Welch spectral analysis
- Filtering price noise with DCT-based low-pass smoothing
- Building robust trading strategies with volatility-based position sizing
- Walk-forward analysis and Monte Carlo validation for strategy robustness
- Portfolio construction with risk parity, volatility scaling, or market cap weighting

## ✨ Features

### 📊 Data & Analysis
- **Data Fetching**: Fetch 30m, 1h, and 4h klines from Binance REST API with automatic retries and rate limiting
- **Smart Caching**: Parquet-based caching with incremental updates and automatic gap detection/backfill
- **DCT Smoothing**: Discrete Cosine Transform-based low-pass smoothing with mirrored padding and tapered cutoff
- **FFT Spectrum Analysis**: Global power spectrum with dominant frequency peaks labeled in bars/hours
- **Sliding Window PSD**: Welch's method for local dominant period extraction over time
- **Spectral Heatmaps**: Time-frequency analysis showing how dominant periods evolve

### 📈 Backtesting & Trading
- **Dynamic Stop Bands**: ATR-based and residual-based stops with configurable multipliers
- **Signal Generation**: Trend-following signals with slope and volatility filters
- **Multi-Timeframe Confirmation**: Execute on 30m with 1h/4h trend filters for higher probability setups
- **Advanced Exits**: Time-based stops, partial take-profit scaling, slope reversal confirmation
- **Dynamic Position Sizing**: Volatility-based (ATR/sigma), fixed risk, optional pyramiding
- **Short/Futures Trading**: Optional short trading mode with configurable fees per venue
- **Vectorized Backtester**: Fast, realistic backtesting with next-bar fills, fees, and slippage
- **Performance Metrics**: 19 metrics including Sharpe, Sortino, win rate, profit factor, and more
- **Trade Analysis**: MAE/MFE tracking, equity curve, complete trade logs with exit reasons

### 🔬 Parameter Optimization (M8)
- **Grid/Random/Bayesian Search**: Multiple optimization algorithms for parameter tuning
- **Walk-Forward Analysis**: Rolling or anchored validation with train/test splits
- **Monte Carlo Resampling**: Block bootstrap for robustness evaluation
- **Rich Visualizations**: Heatmaps, frontier plots, parameter importance, progress tracking
- **Export Capabilities**: CSV/Parquet export with best configurations
- **Reproducible Seeds**: All methods support seeding for reproducibility
- **Batch Processing**: Leaderboard-based evaluation of parameter combinations

### 📊 Portfolio & Risk Management (M9)
- **Multi-Symbol Backtesting**: Parallel per-symbol runs with portfolio aggregation
- **Weighting Schemes**: Equal, volatility-scaled, risk parity, market cap weighted
- **Dynamic Rebalancing**: Configurable frequency and threshold-based rebalancing
- **Correlation Analysis**: Static and rolling correlation matrices
- **Risk Analytics**: Diversification ratio, concentration metrics, risk contributions
- **Exposure Tracking**: Sector exposure and beta calculations
- **Portfolio Metrics**: Comprehensive portfolio-level performance and risk metrics
- **Interactive Portfolio UI**: Symbol basket selection, weight visualization, equity curves

### 🖥️ Infrastructure
- **Interactive UI**: Streamlit-based interface with real-time parameter adjustment
- **Multiple Modes**: Backtesting, Optimization, and Portfolio tabs
- **Preset Management**: Save/load parameter configurations
- **Session Persistence**: Automatic restoration of last session state
- **Live Streaming**: WebSocket-based real-time price updates (optional)
- **Robust Error Handling**: Exponential backoff retries with configurable limits
- **Type Safety**: Full type hints with mypy validation
- **Comprehensive Testing**: 150+ tests covering all components

## 🏗️ Architecture

The platform follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Backtesting  │  │ Optimization │  │  Portfolio   │      │
│  │     Tab      │  │     Tab      │  │     Tab      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                    Core Analysis Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Fourier    │  │   Spectral   │  │   Signals    │      │
│  │  Smoothing   │  │   Analysis   │  │  Generation  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Stop Bands   │  │     MTF      │  │    Exits     │      │
│  │  (ATR/Resid) │  │  Alignment   │  │  Strategies  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐                                            │
│  │   Position   │                                            │
│  │    Sizing    │                                            │
│  └──────────────┘                                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│              Execution & Optimization Layer                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Backtest    │  │ Optimization │  │  Portfolio   │      │
│  │   Engine     │  │  Algorithms  │  │   Manager    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                      Data Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Binance    │  │    Cache     │  │   Loader     │      │
│  │    Client    │  │  (Parquet)   │  │  Unified API │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐                                            │
│  │  WebSocket   │                                            │
│  │  Streaming   │                                            │
│  └──────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

### Architecture Components

#### 1. Data Layer
- **Binance Client** (`core/data/binance_client.py`): REST API client with exponential backoff retries
- **Cache System** (`core/data/cache.py`): Parquet-based storage with gap detection and incremental updates
- **Data Loader** (`core/data/loader.py`): Unified `load_klines()` API abstracting fetch and cache logic
- **WebSocket Streaming** (`core/data/streaming.py`): Real-time kline updates via WebSocket (M6)

#### 2. Core Analysis Modules
- **Fourier Transform** (`core/analysis/fourier.py`): DCT-based smoothing with mirrored padding
- **Spectral Analysis** (`core/analysis/spectral.py`): FFT, Welch PSD, sliding window dominant period
- **Signal Generation** (`core/analysis/signals.py`): Trend-following logic with slope/volatility filters
- **Stop Bands** (`core/analysis/stops.py`): ATR-based and residual-based dynamic stops
- **Multi-Timeframe** (`core/analysis/mtf.py`): Timeframe alignment and trend confirmation (M7)
- **Exit Strategies** (`core/analysis/exits.py`): Time-based, partial profit, slope reversal (M7)
- **Position Sizing** (`core/analysis/sizing.py`): Volatility-based, fixed risk, pyramiding (M7)

#### 3. Execution & Optimization Layer
- **Backtest Engine** (`core/backtest/engine.py`): Vectorized backtesting with realistic fills and fees
- **Optimization** (`core/optimization/`): Grid/Random/Bayesian search, walk-forward, Monte Carlo (M8)
- **Portfolio Manager** (`core/portfolio/`): Multi-symbol execution, weighting, risk analytics (M9)

#### 4. UI Layer
- **Main Tab** (`app/ui/main.py`): Backtesting interface with charts and controls
- **Optimization Tab** (`app/ui/optimization_tab.py`): Parameter tuning and robustness testing (M8)
- **Portfolio Tab** (`app/ui/portfolio_tab.py`): Multi-symbol portfolio management (M9)
- **Live Mode** (`app/ui/live.py`): Real-time streaming and incremental computation (M6)

## 🚀 Tech Stack

- **Language**: Python 3.11+
- **UI Framework**: Streamlit 1.28+
- **Data Processing**: pandas, numpy
- **Analysis**: scipy (FFT, Welch), custom DCT implementation
- **Visualization**: plotly, matplotlib, seaborn
- **HTTP Client**: httpx with tenacity for retries
- **Storage**: Parquet (pyarrow), DuckDB for queries
- **Configuration**: pydantic-settings, python-dotenv
- **WebSocket**: websocket-client
- **Testing**: pytest (150+ tests)
- **Code Quality**: ruff (linting), mypy (type checking)
- **Dependency Management**: Poetry
- **Containerization**: Docker

## 📦 Installation

### Prerequisites

- **Python 3.11 or higher** (check with `python --version`)
- **Poetry** (recommended) or pip
- **Git** for cloning the repository
- **Docker** (optional, for containerized deployment)

### Option 1: Using Poetry (Recommended)

Poetry handles virtual environments and dependency locking automatically.

```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone <repository-url>
cd binance-fourier-backtester

# Install dependencies (creates virtual environment automatically)
poetry install

# Activate virtual environment
poetry shell

# Verify installation
python -c "import streamlit; print('OK')"
```

### Option 2: Using pip

```bash
# Clone the repository
git clone <repository-url>
cd binance-fourier-backtester

# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt  # Create from pyproject.toml if needed

# Verify installation
python -c "import streamlit; print('OK')"
```

### Option 3: Using Docker

```bash
# Clone the repository
git clone <repository-url>
cd binance-fourier-backtester

# Build the Docker image
docker build -t binance-backtester .

# Run the container (mounts data directory for persistence)
docker run -p 8501:8501 -v $(pwd)/data:/app/data binance-backtester

# Access the UI at http://localhost:8501
```

## ⚙️ Configuration

### Environment Variables

Copy the example environment file and customize as needed:

```bash
cp .env.example .env
```

Key configuration options in `.env`:

```bash
# Data paths
BASE_PATH=./data
CACHE_DIR=./data/cache

# Binance API settings
BINANCE_BASE_URL=https://api.binance.com
BINANCE_RATE_LIMIT_PER_MINUTE=1200
BINANCE_REQUEST_TIMEOUT=30

# Trading defaults
DEFAULT_FEE_RATE=0.001          # 0.1% trading fee
DEFAULT_SLIPPAGE_BPS=5.0        # 5 basis points slippage

# Retry settings
MAX_RETRY_ATTEMPTS=5
RETRY_INITIAL_WAIT=1.0          # Initial retry wait (seconds)
RETRY_MAX_WAIT=60.0             # Max retry wait (seconds)

# UI preset storage
PRESET_STORAGE_PATH=./data/presets/presets.yaml
LAST_SESSION_STATE_PATH=./data/presets/last_state.yaml
```

### First-Time Setup

1. **Create data directories** (done automatically on first run):
   ```bash
   mkdir -p data/cache data/presets
   ```

2. **Configure API access** (optional):
   - No API key required for public market data
   - Rate limits apply: 1200 requests/minute default
   - For higher limits, create a Binance account and add API key to `.env`

3. **Test configuration**:
   ```bash
   poetry run python -c "from config.settings import settings; print(f'Cache dir: {settings.cache_dir}')"
   ```

## 🎮 First Run & Usage

### Starting the Application

```bash
# Using Poetry
poetry run streamlit run app/ui/main.py

# Or if in Poetry shell
streamlit run app/ui/main.py

# Using Docker
docker run -p 8501:8501 -v $(pwd)/data:/app/data binance-backtester
```

The UI will open in your browser at `http://localhost:8501`.

### Step-by-Step Usage Guide

#### 1. **Data Loading** (Sidebar)

**Symbol Selection:**
- Choose from supported symbols: BTCUSDT, ETHUSDT, etc.
- Symbols can be extended in `core/data/loader.py`

**Interval Selection:**
- `30m`: 30-minute candles (short-term trading)
- `1h`: 1-hour candles (intraday strategies)
- `4h`: 4-hour candles (swing trading)

**Date Range:**
- Start Date: Beginning of data to fetch (historical data available from 2020)
- End Date: End of data range (up to current date)
- Recommended: Start with 3-6 months for testing

**Load Data Button:**
- First click: Fetches data from Binance API and caches locally
- Subsequent clicks: Uses cached data (fast)
- Force Refresh: Bypass cache and fetch fresh data (slower)

**Data Fetch Process:**
```
Click "Load Data" → Cache Check → API Fetch (if needed) → Gap Detection → Backfill → Display
```

#### 2. **Strategy Parameters** (Sidebar)

**Fourier Smoothing:**
- **Min Trend Period (hours)**: Minimum cycle to preserve
  - Lower values (12-24h): More responsive, more trades
  - Higher values (48-96h): Smoother trend, fewer trades
- **Cutoff Scale**: Smoothing aggressiveness (0.5-2.0)
  - Lower: Less smoothing, follows price closely
  - Higher: More smoothing, ignores short-term noise

**Stop Loss Configuration:**
- **Stop Type**: 
  - `ATR`: Based on Average True Range (volatility)
  - `Residual`: Based on price-trend deviations
- **ATR Period**: Lookback for ATR calculation (14 typical)
- **Residual Window**: Lookback for residual std dev (20 typical)
- **K Stop**: Stop loss multiplier (1.5-3.0)
  - Lower: Tighter stops, more exits
  - Higher: Wider stops, fewer exits
- **K Profit**: Take profit multiplier (2.0-4.0)
  - Should be > K Stop for positive risk/reward

**Signal Generation:**
- **Slope Threshold**: Minimum trend slope for entry (0.0 = any direction)
- **Slope Lookback**: Bars to compute slope (1-5)

**Backtest Configuration:**
- **Initial Capital**: Starting portfolio value ($10,000 typical)
- **Fee Rate**: Trading commission (0.1% = 10 basis points)
- **Slippage**: Expected slippage per trade (0.05% typical)

#### 3. **Running Backtests**

Click **"Run Backtest"** after loading data. The system will:
1. Apply DCT smoothing to price series
2. Compute stop/profit bands
3. Generate entry/exit signals
4. Simulate trades with realistic fills
5. Calculate performance metrics
6. Display results

#### 4. **Understanding Charts**

**Price + Smoothed Trend Chart:**
- **Candlesticks**: OHLC price data
- **Blue line**: DCT-smoothed trend
- **Green markers (▲)**: Long entry signals
- **Red markers (▼)**: Long exit signals
- **Interpretation**: Entries occur when price crosses above trend and slope is positive

**FFT Spectrum:**
- **X-axis**: Frequency (cycles per bar)
- **Y-axis**: Power (amplitude)
- **Peaks**: Dominant cycles in the data
- **Labels**: Period in bars and hours
- **Interpretation**: Identifies major market cycles (e.g., 24h daily cycle, weekly cycles)

**Sliding Dominant Period:**
- **X-axis**: Time (bars)
- **Y-axis**: Dominant period (bars)
- **Line**: Most powerful cycle at each point in time
- **Interpretation**: Shows how market regime changes (trending vs. choppy)

**Welch PSD Heatmap:**
- **X-axis**: Time
- **Y-axis**: Period (bars)
- **Color**: Power density (red = high, blue = low)
- **Interpretation**: Time-frequency map showing cycle evolution

**Equity Curve:**
- **Line**: Portfolio value over time
- **Shaded regions**: Drawdown periods
- **Interpretation**: Visual assessment of strategy performance and risk

#### 5. **Performance Metrics Glossary**

**Returns:**
- `Total Return`: Absolute profit/loss in currency
- `Cumulative Return`: Total % return over period
- `Annualized Return`: % return extrapolated to 1 year

**Risk Metrics:**
- `Max Drawdown`: Largest peak-to-trough decline (%)
- `Max Drawdown $`: Largest peak-to-trough decline in currency
- `Sharpe Ratio`: Risk-adjusted return (return / volatility)
  - < 0: Losing strategy
  - 0-1: Poor to acceptable
  - 1-2: Good
  - \> 2: Excellent
- `Sortino Ratio`: Return / downside deviation (ignores upside volatility)
  - Higher is better, penalizes only downside
- `Calmar Ratio`: Annualized return / max drawdown
  - Measures return per unit of worst-case risk

**Trade Statistics:**
- `N Trades`: Total number of completed round-trips
- `Win Rate`: % of profitable trades
- `Profit Factor`: Gross profit / gross loss
  - < 1.0: Losing overall
  - 1.0-1.5: Marginal
  - 1.5-2.0: Good
  - \> 2.0: Strong
- `Avg Win`: Average profit per winning trade
- `Avg Loss`: Average loss per losing trade
- `Avg Bars Held`: Average trade duration

**Execution Quality:**
- `MAE` (Max Adverse Excursion): Average worst price during trades
- `MFE` (Max Favorable Excursion): Average best price during trades
- `MAE/MFE Analysis`: Helps optimize stop placement

#### 6. **Trade Log & Export**

**Trade Details Table:**
- Entry/Exit times and prices
- P&L per trade
- Duration (bars held)
- Exit reason (stop loss, take profit, signal)

**CSV Download:**
- Click "Download Trades CSV" to export full trade log
- Includes all columns for external analysis
- Compatible with Excel, Python, R

#### 7. **Parameter Presets**

**Saving Presets:**
1. Configure all parameters to your liking
2. Expand "💾 Presets & Persistence"
3. Enter a preset name
4. Click "Save Current Config"
5. Preset is stored in YAML for future use

**Loading Presets:**
1. Expand "💾 Presets & Persistence"
2. Select preset from dropdown
3. Click "Load Selected Preset"
4. All parameters update to saved values

**Session Persistence:**
- Last used parameters automatically saved on exit
- Restored on next app launch
- Disable by clearing `LAST_SESSION_STATE_PATH`

### Multi-Timeframe Strategy (Advanced)

Use the **Multi-Timeframe** section to filter trades:

1. Enable "Use Multi-Timeframe Confirmation"
2. Load higher timeframe data (e.g., 1h or 4h if trading 30m)
3. Set smoothing parameters for each timeframe
4. Trades only execute when all timeframes agree on direction

**Example:**
- Trade on 30m chart
- Confirm with 1h trend: Only long when 1h trend is up
- Confirm with 4h trend: Only long when 4h trend is up
- Result: Fewer but higher-probability trades

### Optimization Tab (M8)

Access via sidebar: **"Optimization"**

**Features:**
- **Grid Search**: Exhaustive search over parameter grid
- **Random Search**: Sample N random combinations
- **Bayesian Optimization**: Intelligent search using Gaussian processes
- **Walk-Forward**: Rolling window validation
- **Monte Carlo**: Block bootstrap resampling for robustness

**Workflow:**
1. Load data (same as main tab)
2. Choose optimization method
3. Define parameter ranges
4. Select objective metric (Sharpe, return, etc.)
5. Click "Run Optimization"
6. View heatmaps, frontier plots, parameter importance
7. Export top configurations to CSV

See [IMPLEMENTATION_M8.md](IMPLEMENTATION_M8.md) for details.

### Portfolio Tab (M9)

Access via sidebar: **"Portfolio"**

**Features:**
- Multi-symbol basket selection
- Weighting methods: Equal, Volatility, Risk Parity, Market Cap
- Correlation matrix and risk analytics
- Portfolio-level metrics and equity curves

**Workflow:**
1. Select multiple symbols from list
2. Choose weighting method
3. Set rebalancing frequency
4. Configure portfolio-level parameters
5. Click "Run Portfolio Backtest"
6. View aggregated results and per-symbol breakdowns

See [IMPLEMENTATION_M9.md](IMPLEMENTATION_M9.md) for details.

## 🐛 Troubleshooting

### Common Errors

#### 1. **Binance Rate Limit Error**

**Error Message:**
```
BinanceRateLimitError: Rate limit exceeded. Please wait 60s.
```

**Solutions:**
- Wait the suggested time before retrying
- Reduce date range to fetch less data
- Use cached data (uncheck "Force Refresh")
- Avoid parallel requests to multiple symbols
- Consider adding Binance API key for higher limits

**Prevention:**
- Let cache populate incrementally over multiple sessions
- Use narrower date ranges initially
- Enable "Force Refresh" only when necessary

#### 2. **Cache Corruption**

**Error Message:**
```
ArrowInvalid: Failed to read parquet file
```

**Solutions:**
- Delete corrupted cache file:
  ```bash
  rm data/cache/BTCUSDT_1h.parquet
  ```
- Or clear entire cache:
  ```bash
  rm -rf data/cache/*
  ```
- Reload data with "Force Refresh" checked

**Prevention:**
- Ensure sufficient disk space
- Avoid manually editing cache files
- Use proper shutdown (Ctrl+C) to avoid incomplete writes

#### 3. **Data Gaps**

**Symptom:** Missing data in charts or unexpected backtest results

**Solutions:**
- Enable "Force Refresh" to trigger gap detection and backfill
- Check Binance API status (may have historical outages)
- Narrow date range to exclude problematic periods

**Detection:**
- System automatically detects gaps > 2x interval duration
- Logs gap detection: Check terminal output

#### 4. **Dependency Conflicts**

**Error Message:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solutions:**
- Ensure virtual environment is activated:
  ```bash
  poetry shell
  ```
- Reinstall dependencies:
  ```bash
  poetry install
  ```
- Check Python version:
  ```bash
  python --version  # Should be 3.11+
  ```

**Poetry Issues:**
- Clear lock file and reinstall:
  ```bash
  rm poetry.lock
  poetry install
  ```

#### 5. **Out of Memory**

**Symptom:** Crashes during large backtests or optimizations

**Solutions:**
- Reduce date range
- Reduce optimization iterations
- Use smaller window lengths for spectral analysis
- Increase system swap space
- Run optimization in batches

**Memory Usage Estimates:**
- 1 year of 1h data: ~50MB
- Grid search (1000 combos): ~500MB-2GB depending on data
- Portfolio with 10 symbols: ~200MB-1GB

#### 6. **Streamlit Port Already in Use**

**Error Message:**
```
Address already in use
```

**Solutions:**
- Kill existing Streamlit process:
  ```bash
  pkill -f streamlit
  ```
- Use different port:
  ```bash
  streamlit run app/ui/main.py --server.port 8502
  ```

#### 7. **WebSocket Connection Failed** (Live Mode)

**Error Message:**
```
WebSocket connection failed
```

**Solutions:**
- Check internet connectivity
- Verify Binance WebSocket endpoint is accessible
- Disable live mode and use historical data only
- Check firewall settings

### Performance Issues

**Slow Data Loading:**
- Normal on first load (API fetch)
- Should be fast on subsequent loads (cache)
- Reduce date range if initial load times out

**Slow Backtest Execution:**
- Normal for large datasets (1+ years)
- Enable vectorized operations (default)
- Reduce spectral analysis window sizes
- Use higher timeframes (4h vs 30m)

**Slow Optimization:**
- Expected for Bayesian (iterative)
- Use Random search for faster results
- Reduce parameter space
- Use walk-forward with fewer folds

### Logging and Debugging

**Enable Debug Logging:**
```python
# In config/settings.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check Cache Status:**
```bash
ls -lh data/cache/
```

**Verify Preset Files:**
```bash
cat data/presets/presets.yaml
cat data/presets/last_state.yaml
```

**Test Data Loading:**
```python
from core.data.loader import load_klines
from datetime import datetime, UTC

df = load_klines(
    symbol="BTCUSDT",
    interval="1h",
    start=datetime(2024, 1, 1, tzinfo=UTC),
    end=datetime(2024, 2, 1, tzinfo=UTC)
)
print(f"Loaded {len(df)} rows")
```

## 📁 Project Structure

```
.
├── app/                            # User interface layer
│   └── ui/
│       ├── main.py                 # Main Streamlit UI with backtesting tab
│       ├── optimization_tab.py     # Parameter optimization interface (M8)
│       ├── portfolio_tab.py        # Portfolio management interface (M9)
│       └── live.py                 # Live streaming coordinator (M6)
│
├── core/                           # Core business logic
│   ├── analysis/                   # Analysis and signal generation
│   │   ├── fourier.py              # DCT-based smoothing functions
│   │   ├── spectral.py             # FFT/Welch PSD analysis and visualization
│   │   ├── signals.py              # Trend-following signal generation
│   │   ├── stops.py                # Dynamic stop loss/take profit bands
│   │   ├── mtf.py                  # Multi-timeframe alignment (M7)
│   │   ├── exits.py                # Advanced exit strategies (M7)
│   │   └── sizing.py               # Position sizing algorithms (M7)
│   │
│   ├── backtest/                   # Backtesting engine
│   │   └── engine.py               # Vectorized backtest with M7 enhancements
│   │
│   ├── data/                       # Data acquisition and management
│   │   ├── binance_client.py       # Binance REST API client
│   │   ├── cache.py                # Parquet caching with gap detection
│   │   ├── loader.py               # Unified data loading API
│   │   ├── streaming.py            # WebSocket streaming (M6)
│   │   └── exceptions.py           # Custom exception types
│   │
│   ├── optimization/               # Parameter optimization framework (M8)
│   │   ├── search.py               # Grid/Random/Bayesian search engines
│   │   ├── params.py               # Parameter space definitions
│   │   ├── walkforward.py          # Walk-forward analysis
│   │   ├── monte_carlo.py          # Monte Carlo resampling
│   │   ├── runner.py               # Optimization orchestration
│   │   └── visualization.py        # Optimization result visualizations
│   │
│   ├── portfolio/                  # Portfolio management (M9)
│   │   ├── portfolio.py            # Main portfolio manager
│   │   ├── weights.py              # Weighting schemes (equal, risk parity, etc.)
│   │   ├── analytics.py            # Risk and correlation analytics
│   │   └── executor.py             # Parallel backtest execution
│   │
│   └── utils/                      # Utility functions
│       └── time.py                 # UTC time utilities
│
├── config/                         # Configuration management
│   ├── settings.py                 # Pydantic settings with .env support
│   └── presets.py                  # Preset and session state management
│
├── tests/                          # Test suite (150+ tests)
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
│
├── examples/                       # Example scripts
│   ├── mtf_strategy_example.py     # Complete MTF strategy example (M7)
│   ├── optimization_example.py     # Optimization workflow example (M8)
│   └── portfolio_example.py        # Portfolio management example (M9)
│
├── docs/                           # Additional documentation
│   ├── ARCHITECTURE.md             # Deep dive into Fourier methods and design
│   ├── CONFIGURATION.md            # All parameters with defaults and ranges
│   └── FAQ.md                      # Frequently asked questions
│
├── IMPLEMENTATION_M7.md            # M7 features documentation
├── IMPLEMENTATION_M8.md            # M8 features documentation
├── IMPLEMENTATION_M9.md            # M9 features documentation
├── IMPLEMENTATION_SUMMARY.md       # M1 implementation summary
├── M7_SUMMARY.md                   # M7 milestone summary
├── M9_SUMMARY.md                   # M9 milestone summary
│
├── Dockerfile                      # Docker container configuration
├── docker-compose.yml              # Docker Compose setup (if exists)
├── pyproject.toml                  # Poetry dependencies and tool config
├── requirements.txt                # Pip dependencies (generated from pyproject.toml)
├── .env.example                    # Environment variable template
├── .gitignore                      # Git ignore rules
├── .ruff.toml                      # Ruff linter configuration
├── mypy.ini                        # Mypy type checker configuration
└── README.md                       # This file
```

### Module Explanations

**Core Modules:**
- `fourier.py`: Implements DCT-based low-pass filtering with mirrored padding to avoid edge effects
- `spectral.py`: FFT and Welch PSD for frequency analysis, identifies dominant market cycles
- `signals.py`: Trend-following entry/exit logic with slope and volatility filtering
- `stops.py`: Computes ATR-based (volatility) and residual-based (deviation) stop bands
- `engine.py`: Vectorized backtester simulates fills at next bar open with fees and slippage

**Advanced Features:**
- `mtf.py`: Aligns multiple timeframes and checks trend agreement
- `exits.py`: Time-based exits, partial profit taking, slope reversal confirmation
- `sizing.py`: Volatility-scaled position sizing to normalize risk per trade
- `optimization/`: Parameter tuning framework with multiple search algorithms
- `portfolio/`: Multi-symbol management with risk analytics and rebalancing

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=core --cov=config --cov=app

# Run specific test file
poetry run pytest tests/test_backtest.py

# Run with verbose output
poetry run pytest -v

# Run only optimization tests
poetry run pytest tests/test_optimization.py

# Run with markers (if defined)
poetry run pytest -m "not slow"
```

### Test Coverage

- **150+ tests** across all modules
- **Core analysis**: Fourier, spectral, signals, stops
- **Backtesting**: Engine logic, enhanced features
- **Data layer**: API client, caching, gap detection
- **Optimization**: Search algorithms, walk-forward, Monte Carlo
- **Portfolio**: Weighting, analytics, execution
- **Integration**: End-to-end strategy workflows

## 🛠️ Development

### Code Quality Tools

```bash
# Format code (auto-fix)
poetry run ruff format .

# Lint code (auto-fix where possible)
poetry run ruff check . --fix

# Lint without auto-fix
poetry run ruff check .

# Type check
poetry run mypy .

# Run all checks
poetry run ruff format . && poetry run ruff check . && poetry run mypy . && poetry run pytest
```

### Pre-commit Setup (Optional)

```bash
# Install pre-commit
poetry add --group dev pre-commit

# Install hooks
poetry run pre-commit install

# Run manually
poetry run pre-commit run --all-files
```

### Adding New Symbols

Edit `core/data/loader.py`:

```python
SUPPORTED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"}  # Add symbols here
```

### Adding New Intervals

Edit `core/data/loader.py`:

```python
SUPPORTED_INTERVALS = {"30m", "1h", "4h", "1d"}  # Add intervals here
```

Ensure Binance API supports the interval.

### Extending Strategies

Create new signal generators in `core/analysis/signals.py`:

```python
def generate_mean_reversion_signals(
    close: np.ndarray,
    smoothed: np.ndarray,
    threshold: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """Generate mean reversion signals."""
    deviation = (close - smoothed) / smoothed
    entries = deviation < -threshold  # Buy when oversold
    exits = deviation > threshold     # Sell when overbought
    return entries, exits
```

Integrate into UI in `app/ui/main.py`.

## 📚 API Reference

For detailed API documentation, see:
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: In-depth technical design and algorithms
- **[CONFIGURATION.md](docs/CONFIGURATION.md)**: Complete parameter reference with ranges
- **[FAQ.md](docs/FAQ.md)**: Common questions and answers

### Quick API Example

```python
from datetime import datetime, UTC
from core.data.loader import load_klines
from core.analysis.fourier import smooth_price_series
from core.analysis.stops import compute_atr_stops
from core.analysis.signals import generate_signals_with_stops
from core.backtest.engine import BacktestConfig, run_backtest

# Load data
df = load_klines(
    symbol="BTCUSDT",
    interval="1h",
    start=datetime(2024, 1, 1, tzinfo=UTC),
    end=datetime(2024, 6, 1, tzinfo=UTC)
)

# Extract OHLCV
close = df["close"].values
high = df["high"].values
low = df["low"].values
open_prices = df["open"].values
timestamps = df["open_time"]

# Smooth prices
smoothed = smooth_price_series(
    prices=close,
    min_period_bars=24,  # 24-hour trend
    cutoff_scale=1.0
)

# Compute stop bands
long_stop, long_profit, _, _ = compute_atr_stops(
    close=close,
    high=high,
    low=low,
    atr_period=14,
    k_stop=2.0,
    k_profit=3.0
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
    initial_capital=10000.0,
    fee_rate=0.001,
    slippage=0.0005
)

result = run_backtest(
    signals=signals,
    open_prices=open_prices,
    high_prices=high,
    low_prices=low,
    close_prices=close,
    timestamps=timestamps,
    config=config
)

# Print results
print(f"Total Return: {result.metrics['total_return']:.2f}")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Win Rate: {result.metrics['win_rate']:.2%}")
print(f"Max Drawdown: {result.metrics['max_drawdown_pct']:.2%}")
print(f"Number of Trades: {result.metrics['n_trades']}")
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make changes** following existing code style
4. **Run tests**: `poetry run pytest`
5. **Run linting**: `poetry run ruff check .`
6. **Run type checking**: `poetry run mypy .`
7. **Commit changes**: `git commit -m "Add feature"`
8. **Push to branch**: `git push origin feature/your-feature`
9. **Open Pull Request**

### Code Style Guidelines

- Follow PEP 8 with 100-character line limit
- Use type hints for all function signatures
- Write docstrings for public functions
- Add tests for new features
- Keep functions focused and modular
- Use descriptive variable names

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat: Add Bollinger Bands signal generator

- Implement standard deviation bands
- Add integration with existing signal pipeline
- Include tests for edge cases

Closes #123
```

## 📄 License

MIT License

Copyright (c) 2024 Binance Fourier Backtester Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 🙏 Acknowledgments

- **Binance** for providing free market data API
- **Streamlit** for the excellent UI framework
- **SciPy** for robust FFT and signal processing tools
- **Poetry** for modern Python dependency management

## 📞 Support & Resources

- **Documentation**: See [docs/](docs/) folder
- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions
- **Examples**: See [examples/](examples/) folder for complete workflows

## 🗺️ Roadmap

**Completed:**
- ✅ M1: Data layer and caching
- ✅ M2: Fourier analysis and smoothing
- ✅ M3: Basic backtesting engine
- ✅ M4: Visualization and UI
- ✅ M5: Dynamic stops and signals
- ✅ M6: Live streaming (optional)
- ✅ M7: Multi-timeframe and advanced features
- ✅ M8: Parameter optimization
- ✅ M9: Portfolio management

**Future Enhancements:**
- [ ] Machine learning integration (feature engineering from Fourier)
- [ ] Options strategies (straddles, spreads)
- [ ] Automated trade execution via Binance API
- [ ] Telegram/Discord notifications
- [ ] Custom indicator library
- [ ] Multi-exchange support (FTX, Coinbase)
- [ ] Real-time risk monitoring dashboard

---

**Happy Trading! 🚀📈**

For detailed technical documentation, see:
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design and algorithms
- [CONFIGURATION.md](docs/CONFIGURATION.md) - Complete parameter reference
- [FAQ.md](docs/FAQ.md) - Frequently asked questions
