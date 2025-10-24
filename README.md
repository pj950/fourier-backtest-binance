# Binance Fourier Backtester

A Python 3.11 + Streamlit application for fetching, caching, and visualizing Binance OHLCV data with support for incremental updates and gap detection.

## Features

- **Data Fetching**: Fetch 30m, 1h, and 4h klines from Binance REST API with automatic retries and rate limiting
- **Smart Caching**: Parquet-based caching with incremental updates and automatic gap detection/backfill
- **Interactive UI**: Streamlit-based interface with candlestick charts, volume bars, and data exploration
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
│   ├── data/
│   │   ├── binance_client.py    # Binance API client
│   │   ├── cache.py             # Parquet caching with gap detection
│   │   └── loader.py            # Unified data loading API
│   └── utils/
│       └── time.py              # UTC time utilities
├── config/
│   └── settings.py              # Configuration management
├── tests/
│   └── test_data_fetch_cache.py # Unit tests
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

5. The first load will fetch historical data from 2020-01-01. Subsequent loads use cached data with incremental updates.

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

### `load_klines(symbol, interval, start, end, force_refresh=False)`

Load OHLCV data for a given symbol and time range.

**Parameters:**
- `symbol` (str): Trading pair (e.g., "BTCUSDT")
- `interval` (str): Timeframe ("30m", "1h", "4h")
- `start` (datetime): Start time (UTC)
- `end` (datetime): End time (UTC)
- `force_refresh` (bool): Bypass cache and fetch fresh data

**Returns:**
- `pd.DataFrame`: OHLCV data with columns: open_time, open, high, low, close, volume, quote_volume, trades, close_time

## License

MIT License
