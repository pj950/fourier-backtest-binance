# M7 Implementation: Strategy Enhancements

## Overview

M7 adds advanced strategy features including multi-timeframe confirmation, additional exit types, dynamic position sizing, and optional short/futures trading.

## Features Implemented

### 1. Multi-Timeframe (MTF) Confirmation

**Module**: `core/analysis/mtf.py`

Execute trades on 30m timeframe while filtering with 1h and 4h trend alignment.

#### Key Functions

- `align_timeframes()`: Align higher timeframe data to lower timeframe bars
- `compute_trend_direction()`: Determine trend direction (1=up, -1=down, 0=neutral)
- `check_mtf_alignment()`: Check if trends align across multiple timeframes
- `apply_mtf_filter()`: Filter entry signals by MTF alignment

#### Example Usage

```python
from core.analysis.mtf import (
    align_timeframes,
    compute_trend_direction,
    check_mtf_alignment,
    apply_mtf_filter,
)

# Load multiple timeframes
df_30m = load_klines("BTCUSDT", "30m", start, end)
df_1h = load_klines("BTCUSDT", "1h", start, end)
df_4h = load_klines("BTCUSDT", "4h", start, end)

# Align timeframes
df_aligned = align_timeframes(df_30m, df_1h, "30m", "1h")
df_aligned = align_timeframes(df_aligned, df_4h, "30m", "4h")

# Compute trends for each timeframe
trend_30m = compute_trend_direction(close, smoothed, slope_lookback=2)
trend_1h = compute_trend_direction(close, smoothed_1h, slope_lookback=2)
trend_4h = compute_trend_direction(close, smoothed_4h, slope_lookback=2)

# Check alignment (require all 3 or at least 2 of 3)
aligned_long, aligned_short = check_mtf_alignment(
    trend_30m, trend_1h, trend_4h, require_all=True
)

# Filter signals
signals_filtered = apply_mtf_filter(signals, aligned_long, direction=1)
```

### 2. Additional Exit Strategies

**Module**: `core/analysis/exits.py`

#### Time-Based Exit

Exit after holding position for maximum number of bars.

```python
from core.analysis.exits import check_time_based_exit

if check_time_based_exit(entry_idx, current_idx, max_bars_held=50):
    # Exit position
    pass
```

#### Partial Take-Profit

Scale out of positions at multiple price levels.

```python
from core.analysis.exits import compute_partial_tp_levels, check_partial_tp_hit

# Define scales: (price_move%, position_size%)
scales = [(0.02, 0.5), (0.05, 0.3), (0.10, 0.2)]

tp_levels = compute_partial_tp_levels(
    entry_price=100.0,
    direction=1,  # 1 for long, -1 for short
    scales=scales,
)

# Check which levels hit
hit_levels = set()
newly_hit = check_partial_tp_hit(
    current_price, high_price, low_price,
    tp_levels, hit_levels, direction=1
)
```

#### Slope Reversal Confirmation

Exit when smoothed trend slope reverses.

```python
from core.analysis.exits import compute_slope_reversal

reversal = compute_slope_reversal(
    smoothed, lookback=2, threshold=0.5
)

# Exit when reversal[i] == True
```

### 3. Dynamic Position Sizing

**Module**: `core/analysis/sizing.py`

#### Volatility-Based Sizing

Size positions based on ATR or sigma targeting.

```python
from core.analysis.sizing import compute_volatility_based_size

size = compute_volatility_based_size(
    capital=10000.0,
    entry_price=100.0,
    stop_price=95.0,
    volatility=2.0,  # ATR or sigma
    risk_target=0.02,  # Target 2% volatility exposure
    max_risk_per_trade=0.05,  # Max 5% risk per trade
    vol_target_mode="atr",  # or "sigma"
)
```

#### Fixed Risk Sizing

Size based on fixed percentage risk per trade.

```python
from core.analysis.sizing import compute_fixed_risk_size

size = compute_fixed_risk_size(
    capital=10000.0,
    entry_price=100.0,
    stop_price=95.0,
    risk_fraction=0.01,  # Risk 1% per trade
)
```

#### Pyramiding (Optional)

Add to winning positions.

```python
from core.analysis.sizing import compute_pyramid_size, check_pyramid_conditions

if check_pyramid_conditions(entry_price, current_price, profit_threshold=0.03, direction=1):
    additional_size = compute_pyramid_size(
        initial_size=10.0,
        current_position=10.0,
        max_pyramids=3,
        pyramid_scale=0.5,  # Each add is 50% of previous
    )
```

### 4. Enhanced Backtest Engine

**Module**: `core/backtest/engine.py`

#### New BacktestConfig Parameters

```python
@dataclass
class BacktestConfig:
    # Existing parameters
    initial_capital: float = 10000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    position_size_mode: str = "full"
    position_size_fraction: float = 1.0
    
    # NEW: Short/futures trading
    allow_shorts: bool = False
    
    # NEW: Time-based exits
    max_bars_held: int | None = None
    
    # NEW: Partial take-profit
    enable_partial_tp: bool = False
    partial_tp_scales: list[tuple[float, float]] | None = None
    
    # NEW: Pyramiding
    enable_pyramiding: bool = False
    max_pyramids: int = 3
    pyramid_scale: float = 0.5
    
    # NEW: Dynamic sizing
    sizing_mode: str = "fixed"  # "fixed", "volatility", "fixed_risk"
    volatility_target: float = 0.02
    max_risk_per_trade: float = 0.02
```

#### New Trade Fields

```python
@dataclass
class Trade:
    # Existing fields
    entry_idx: int
    exit_idx: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    mae: float
    mfe: float
    mae_pct: float
    mfe_pct: float
    bars_held: int
    fees: float
    
    # NEW fields
    direction: int = 1  # 1 for long, -1 for short
    exit_reason: str = "signal"  # "signal", "stop", "time"
    partial_exits: list[tuple[int, float, float]] | None = None
```

#### Enhanced Backtest Function

```python
from core.backtest.engine import run_backtest_enhanced, BacktestConfig

config = BacktestConfig(
    initial_capital=10000.0,
    allow_shorts=True,
    max_bars_held=100,
    sizing_mode="volatility",
    volatility_target=0.02,
    max_risk_per_trade=0.02,
)

result = run_backtest_enhanced(
    signals=signals,
    open_prices=open_prices,
    high_prices=high_prices,
    low_prices=low_prices,
    close_prices=close_prices,
    timestamps=timestamps,
    atr=atr,  # For volatility sizing
    stop_levels=stop_levels,  # For stop loss
    config=config,
)
```

#### Signal Convention for Enhanced Backtest

- `1`: Enter long at next bar open
- `-1`: Exit long position (or enter short if `allow_shorts=True`)
- `0`: Hold current position
- `2`: Enter short at next bar open (only if `allow_shorts=True`)
- `-2`: Exit short position

### 5. Short/Futures Mode (Optional)

Enable short trading with the `allow_shorts` flag:

```python
config = BacktestConfig(
    initial_capital=10000.0,
    allow_shorts=True,
    fee_rate=0.0004,  # Futures fee rate (can be different from spot)
)
```

When shorts are enabled:
- Signal `2` enters short positions
- Signal `-2` exits short positions
- Different fee rates can be configured per venue
- Short positions are tracked with negative position size
- PnL calculation accounts for short mechanics

## Testing

Comprehensive test coverage includes:

- **test_mtf.py**: Multi-timeframe alignment and filtering
- **test_exits.py**: Time-based, partial TP, slope reversal exits
- **test_sizing.py**: Volatility-based, fixed risk, pyramiding sizing
- **test_backtest_enhanced.py**: Enhanced backtest engine functionality
- **test_strategy_integration.py**: Full strategy integration tests

Run tests:
```bash
pytest tests/test_mtf.py -v
pytest tests/test_exits.py -v
pytest tests/test_sizing.py -v
pytest tests/test_backtest_enhanced.py -v
pytest tests/test_strategy_integration.py -v
```

## Complete Example

See `examples/mtf_strategy_example.py` for a complete working example that demonstrates all M7 features.

Run the example:
```bash
python examples/mtf_strategy_example.py
```

## Configuration Parameters

### Multi-Timeframe Settings

- **Execution Timeframe**: 30m (default)
- **Trend Filter 1**: 1h
- **Trend Filter 2**: 4h
- **Alignment Mode**: All agree or 2-of-3 majority

### Exit Settings

- **Time-Based**: `max_bars_held` (e.g., 50, 100, 200)
- **Partial TP**: List of `(price_pct, size_pct)` tuples
  - Example: `[(0.02, 0.5), (0.05, 0.3), (0.10, 0.2)]`
- **Slope Reversal**: `lookback` and `threshold` parameters

### Sizing Settings

- **Mode**: "fixed", "volatility", or "fixed_risk"
- **Volatility Target**: 0.01-0.05 (1%-5% of capital)
- **Max Risk Per Trade**: 0.01-0.03 (1%-3% of capital)

### Futures Settings

- **Allow Shorts**: `True`/`False`
- **Fee Rate**: 0.0004 for futures (vs 0.001 for spot)
- **Slippage**: Can be adjusted per venue

## Performance Considerations

1. **MTF Alignment**: Requires loading and aligning multiple timeframes
2. **Partial TP**: Adds complexity to trade tracking
3. **Dynamic Sizing**: Requires ATR or volatility calculation per bar
4. **Vectorized Operations**: All calculations use NumPy for performance

## Metrics and Logging

The enhanced backtest tracks:
- Direction of each trade (long/short)
- Exit reason (signal/stop/time)
- Partial exit history
- Position sizing decisions
- MTF alignment status

All standard metrics remain available:
- Total/annualized returns
- Sharpe/Sortino ratios
- Win rate, profit factor
- MAE/MFE analysis
- Drawdown statistics

## Future Enhancements

Potential additions for future milestones:
- Dynamic fee rates per venue
- More sophisticated pyramiding logic
- Trailing partial take-profit
- Correlation-based position sizing
- Multi-asset portfolio backtesting
