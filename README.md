# Binance Fourier Backtester

A Python 3.11 + Streamlit application for fetching, caching, and visualizing Binance OHLCV data with advanced Fourier analysis and spectral smoothing capabilities.

## Features

- **Data Fetching**: Fetch 30m, 1h, and 4h klines from Binance REST API with automatic retries and rate limiting
- **Smart Caching**: Parquet-based caching with incremental updates and automatic gap detection/backfill
- **DCT Smoothing**: Discrete Cosine Transform-based low-pass smoothing with mirrored padding and tapered cutoff
- **FFT Spectrum Analysis**: Global power spectrum with dominant frequency peaks labeled in bars/hours
- **Sliding Window PSD**: Welch's method for local dominant period extraction over time
- **Spectral Heatmaps**: Time-frequency analysis showing how dominant periods evolve
- **Interactive UI**: Streamlit-based interface with real-time parameter adjustment
- **Robust Error Handling**: Exponential backoff retries with configurable limits
- **Type Safety**: Full type hints with mypy validation

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

## Project Structure

```
.
├── app/
│   └── ui/
│       └── main.py              # Streamlit UI entry point
├── core/
│   ├── analysis/
│   │   ├── fourier.py           # DCT-based smoothing functions
│   │   └── spectral.py          # FFT/Welch PSD analysis
│   ├── data/
│   │   ├── binance_client.py    # Binance API client
│   │   ├── cache.py             # Parquet caching with gap detection
│   │   └── loader.py            # Unified data loading API
│   └── utils/
│       └── time.py              # UTC time utilities
├── config/
│   └── settings.py              # Configuration management
├── tests/
│   ├── test_data_fetch_cache.py # Data layer tests
│   ├── test_fourier.py          # Fourier smoothing tests
│   └── test_spectral.py         # Spectral analysis tests
├── Dockerfile                   # Docker configuration
├── pyproject.toml              # Poetry dependencies
└── README.md                   # This file
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

## License

MIT License
