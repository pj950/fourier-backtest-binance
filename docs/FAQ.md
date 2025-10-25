# Frequently Asked Questions (FAQ)

Common questions and answers about the Binance Fourier Backtester.

## Table of Contents

1. [General Questions](#general-questions)
2. [Data & Caching](#data--caching)
3. [Fourier Analysis](#fourier-analysis)
4. [Backtesting](#backtesting)
5. [Optimization](#optimization)
6. [Portfolio Management](#portfolio-management)
7. [Performance & Troubleshooting](#performance--troubleshooting)

---

## General Questions

### What is this platform for?

The Binance Fourier Backtester is designed for quantitative traders and researchers to:
- Analyze cryptocurrency price data using advanced signal processing techniques
- Develop and backtest trend-following trading strategies
- Optimize strategy parameters using multiple algorithms
- Manage multi-symbol portfolios with risk controls

### Do I need coding experience to use it?

**Basic usage**: No. The Streamlit UI provides a point-and-click interface for:
- Loading data
- Running backtests
- Viewing results
- Exporting trade logs

**Advanced usage**: Yes. For:
- Custom strategy development
- API-level integration
- Extending the codebase
- Automated execution

### Is this production-ready for live trading?

**Short answer**: No, it's a research and backtesting platform.

**Long answer**: While the backtesting engine is robust and realistic, live trading requires:
- Real-time order execution infrastructure
- Risk management systems
- Monitoring and alerting
- Regulatory compliance
- Capital at risk

Use this platform for strategy development and validation, then implement production systems separately.

### What exchanges are supported?

Currently: **Binance only** (spot markets via REST API).

**Future support planned**:
- Binance futures
- Other major exchanges (Coinbase, Kraken, etc.)
- DeFi protocols (Uniswap via The Graph)

### What cryptocurrencies can I trade?

Any Binance spot pair. Default supported:
- BTCUSDT
- ETHUSDT
- BNBUSDT
- ADAUSDT

**To add more**: Edit `SUPPORTED_SYMBOLS` in `core/data/loader.py`.

---

## Data & Caching

### How far back does historical data go?

**Binance availability**:
- Bitcoin (BTC): Back to 2017
- Most major alts: 2018-2020
- Newer listings: From listing date

**Platform limitation**: None, limited only by Binance API.

### How much disk space does cache use?

**Estimates** (Parquet compressed):
| Interval | Duration | Approx Size |
|----------|----------|------------|
| 30m | 1 year | ~5 MB |
| 1h | 1 year | ~2.5 MB |
| 4h | 1 year | ~0.6 MB |

**Example**: 5 symbols, 1h data, 2 years = ~25 MB total.

**Storage**: Very efficient thanks to Parquet columnar compression.

### Do I need to manually update cached data?

**No**. The cache system automatically:
1. Detects that you have old data
2. Fetches only new bars since last update
3. Appends to cache
4. Handles gaps and deduplication

**Manual refresh**: Only use "Force Refresh" if you suspect corruption.

### What if my cached data gets corrupted?

**Symptoms**:
- Error loading cache file
- Missing data in charts
- Inconsistent timestamps

**Solution**:
```bash
# Delete corrupted file
rm data/cache/BTCUSDT_1h.parquet

# Or delete all cache
rm -rf data/cache/*

# Reload with Force Refresh checked
```

### Can I use data from other sources (CSV, databases)?

**Not directly**, but you can:

1. Load your data into pandas DataFrame with required columns:
   ```python
   # Required columns: open_time, open, high, low, close, volume
   df = pd.read_csv("my_data.csv")
   ```

2. Convert timestamps to pandas datetime:
   ```python
   df['open_time'] = pd.to_datetime(df['timestamp'])
   ```

3. Pass to backtest engine directly (bypass loader).

**Note**: No UI support currently, requires Python scripting.

---

## Fourier Analysis

### Why use Fourier transforms for trading?

**Traditional smoothing** (moving averages):
- Lag: Always lag price by (period/2)
- Fixed window: Doesn't adapt to market regime
- Endpoints: Poor behavior at boundaries

**Fourier smoothing**:
- Frequency-based: Removes noise while preserving trend
- Adaptive: Automatically handles different cycle lengths
- Endpoints: Mirrored padding avoids edge distortions

**Result**: Cleaner trend signals with less lag and fewer false breakouts.

### What does "30m聚合" mean in the context?

**Context**: Chinese term meaning "30-minute aggregation".

**Issue**: How to handle 30-minute bars at boundaries?
- OHLCV data must align to :00 or :30 timestamps
- Partial bars at edges are discarded or padded

**Solution in platform**: 
- Data loader enforces timestamp alignment
- Partial bars at query boundaries excluded
- Ensures consistent bar spacing for FFT

### Why DCT instead of FFT?

**FFT (Fast Fourier Transform)**:
- Assumes signal is periodic
- Creates discontinuity at boundaries (repeats signal)
- Complex-valued output

**DCT (Discrete Cosine Transform)**:
- Assumes signal is reflected (mirrored) at boundaries
- Smoother transition, no discontinuity
- Real-valued output (easier interpretation)

**For financial data**: DCT is better because price is not periodic.

**Visualization**:
```
FFT boundary:  ...x[n-1], x[n], x[0], x[1]...  (jump!)
DCT boundary:  ...x[n-1], x[n], x[n], x[n-1]... (smooth)
```

### What are "dominant peaks" in FFT spectrum?

**Definition**: Frequencies with highest power in the spectrum.

**Interpretation**: Major cycles in the price data.

**Example**:
- Peak at 24 bars: Daily cycle (24h on 1h data)
- Peak at 168 bars: Weekly cycle
- Peak at 12 bars: Half-day cycle

**Use case**: 
- Identify market rhythm
- Set `min_trend_hours` near dominant period
- Understand regime (trending vs. cyclical)

### How does spectral heatmap help?

**What it shows**: Time-frequency representation.
- X-axis: Time (progression through dataset)
- Y-axis: Period (cycle length in bars)
- Color: Power (red = strong cycle, blue = weak)

**What you can see**:
- Regime changes: Shift from short to long cycles (or vice versa)
- Cycle consistency: Vertical bands = stable cycle
- Cycle instability: Scattered colors = no dominant cycle

**Trading insight**:
- Strong vertical band: Predictable market, trend-following works
- Scattered pattern: Unpredictable, consider mean reversion or stay out

### What's the relationship between period and frequency?

**Mathematical**:
```
frequency = 1 / period
period = 1 / frequency
```

**Example** (1h interval):
- Period = 24 bars → Frequency = 1/24 ≈ 0.042 cycles/bar
- Period = 168 bars → Frequency = 1/168 ≈ 0.006 cycles/bar

**Cutoff frequency**:
- Cutoff freq = 1 / min_period_bars
- Preserves cycles longer than `min_period`
- Removes cycles shorter than `min_period`

---

## Backtesting

### Are the backtest results realistic?

**Realistic aspects**:
- ✅ Next-bar fills (no look-ahead bias)
- ✅ Fees modeled (entry + exit)
- ✅ Slippage modeled
- ✅ Stop loss checked on intrabar highs/lows
- ✅ Single position at a time (no over-trading)

**Simplifications**:
- ⚠️ Assumes infinite liquidity (can always fill at open price)
- ⚠️ No partial fills
- ⚠️ No margin calls or liquidations
- ⚠️ No funding rates (for futures)

**For most retail strategies**: Results are realistic.

**For large positions**: Manually increase slippage to account for market impact.

### What's the difference between signals and fills?

**Signal**: Indicator that strategy wants to enter/exit.
- Generated at bar **close**
- Can't trade at close price (you learn close only after bar closes)

**Fill**: Actual execution of trade.
- Occurs at **next bar open**
- Price = open[t+1] + slippage

**Example timeline**:
```
Bar 100 close: Signal generated (close > smoothed)
Bar 101 open:  Fill executed (buy at open price)
```

**Why important**: Prevents look-ahead bias (trading on information you don't have yet).

### Why do I get different results on same data?

**Should NOT happen** if:
- Same data (cached)
- Same parameters
- Same random seed (for optimizations)

**Might happen** if:
- Data updated (Force Refresh adds new bars)
- Different slippage/fee settings
- Different parameter values
- Optimization methods with randomness (if seed not set)

**To ensure reproducibility**:
1. Save preset before running
2. Note date range used
3. Set random seed for optimizations
4. Export results immediately

### What's look-ahead bias and how do you avoid it?

**Look-ahead bias**: Using future information to make past decisions.

**Example of bias**:
```python
# WRONG: Uses close[t] to trade at close[t]
if close[t] > smoothed[t]:
    buy_at(close[t])  # Can't know close[t] until bar closes!
```

**Correct approach**:
```python
# Correct: Uses close[t] to trade at open[t+1]
if close[t] > smoothed[t]:
    signal[t] = 1
    # Fill at open[t+1]
    buy_at(open[t+1])
```

**In this platform**: Enforced by design. Signals generated at bar close, fills at next bar open.

### What do MAE and MFE mean?

**MAE (Maximum Adverse Excursion)**:
- Worst point in trade (maximum loss during trade)
- Measures "how wrong you were"
- Use to optimize stop placement

**MFE (Max Favorable Excursion)**:
- Best point in trade (maximum profit during trade)
- Measures "how right you were"
- Use to optimize profit taking

**Example**:
```
Entry: $40,000
Exit: $41,000 (+2.5%)
Lowest price during trade: $39,500 (-1.25%) ← MAE
Highest price during trade: $42,000 (+5%) ← MFE
```

**Interpretation**:
- High MAE: Stop too wide or bad timing
- High MFE: Left profit on table, consider wider profit target

### How are Sharpe and Sortino different?

**Sharpe Ratio**:
```
Sharpe = (Return - Risk Free Rate) / Volatility
```
- Penalizes both upside and downside volatility
- Good for symmetric risk

**Sortino Ratio**:
```
Sortino = (Return - Risk Free Rate) / Downside Deviation
```
- Penalizes only downside volatility
- Better for asymmetric strategies (e.g., options, trend-following)

**When Sortino > Sharpe**: Strategy has more upside volatility (good!).

**Example**:
- Strategy A: 20% return, 10% vol, 5% downside vol
  - Sharpe: 20/10 = 2.0
  - Sortino: 20/5 = 4.0
- Strategy has positive skew (big wins, small losses)

---

## Optimization

### How do I choose between Grid, Random, and Bayesian?

**Grid Search**:
- **When**: Small parameter space (< 1000 combos)
- **Pros**: Exhaustive, easy to visualize
- **Cons**: Exponential growth, slow for high dimensions
- **Use case**: 2-3 parameters with 5-10 values each

**Random Search**:
- **When**: Large parameter space, quick results
- **Pros**: Fast, often finds near-optimal
- **Cons**: No learning, may miss optimal
- **Use case**: 5+ parameters, want results in < 1 hour

**Bayesian Optimization**:
- **When**: Expensive backtest, continuous parameters
- **Pros**: Intelligent search, fewer iterations needed
- **Cons**: Slow per iteration, needs more initial samples
- **Use case**: Computationally expensive strategies, need best results

**Rule of thumb**:
- < 100 combos → Grid
- 100-1000 combos → Random (n_iter=200-500)
- > 1000 combos or continuous → Bayesian (n_iter=50-100)

### What's overfitting and how do I avoid it?

**Overfitting**: Parameters work great on historical data but fail on new data.

**Causes**:
1. Too many parameters
2. Too little data
3. No out-of-sample validation
4. Optimizing for noise instead of signal

**Symptoms**:
- Perfect in-sample results (Sharpe > 5)
- Poor out-of-sample results
- Strategy fails immediately after optimization

**Prevention**:
1. **Walk-forward analysis**: Always validate on unseen data
2. **Parameter parsimony**: Fewer parameters = more robust
3. **Longer data periods**: Optimize on 2+ years
4. **Realistic transaction costs**: Don't optimize with zero fees
5. **Monte Carlo testing**: Ensure results aren't due to lucky sequence

**Golden rule**: If IS/OOS Sharpe ratio > 2.0, you're probably overfit.

### What's walk-forward analysis?

**Concept**: Simulate realistic strategy deployment.

**Process**:
1. Split data into windows (e.g., 6 months train, 2 months test)
2. Optimize on train window
3. Test on following test window (out-of-sample)
4. Roll forward, repeat
5. Aggregate OOS results

**Example** (2 years of data):
```
Window 1: Train Jan-Jun 2023, Test Jul-Aug 2023
Window 2: Train Jul-Dec 2023, Test Jan-Feb 2024
Window 3: Train Jan-Jun 2024, Test Jul-Aug 2024
...
```

**Evaluation**:
- If OOS performance ~similar to IS: Robust strategy ✅
- If OOS << IS: Overfitting ❌
- If OOS > IS: Lucky, or regime change (investigate)

### How long should train/test windows be?

**Rules of thumb**:

**Train window**:
- Minimum: 3 months (90 days)
- Recommended: 6-12 months
- Maximum: 80% of total data

**Test window**:
- Minimum: 1 month (30 days)
- Recommended: 2-3 months
- Maximum: 20% of total data

**Ratio**: Train:Test should be 3:1 to 5:1.

**Example setups**:
| Total Data | Train | Test | Windows |
|-----------|-------|------|---------|
| 1 year | 6 months | 2 months | 3 |
| 2 years | 6 months | 2 months | 6 |
| 2 years | 12 months | 3 months | 4 |

---

## Portfolio Management

### How do I choose a weighting method?

**Equal Weight**:
- **Best for**: Maximum diversification by count
- **Pros**: Simple, no optimization
- **Cons**: Ignores risk differences
- **Use case**: Large basket (20+ symbols), no strong views

**Inverse Volatility**:
- **Best for**: Normalizing risk across assets
- **Pros**: Fast, intuitive
- **Cons**: Can underweight good performers
- **Use case**: Mixed volatility assets (BTC + altcoins)

**Risk Parity**:
- **Best for**: Maximum risk diversification
- **Pros**: Equal risk contribution from each asset
- **Cons**: Requires optimization (slow), needs covariance matrix
- **Use case**: Long-term portfolios, low turnover

**Market Cap**:
- **Best for**: Index tracking
- **Pros**: Momentum bias (big = strong)
- **Cons**: Concentration risk (BTC dominates)
- **Use case**: Passive crypto index

**Recommendation**: Start with inverse volatility (good balance of simplicity and effectiveness).

### How often should I rebalance?

**Frequency options**:
1. **Daily**: Maintains target weights precisely
   - Pros: Always aligned
   - Cons: High transaction costs
   
2. **Weekly**: Common for active strategies
   - Pros: Balance of control and costs
   - Cons: Some drift between rebalances
   
3. **Monthly**: Typical for long-term portfolios
   - Pros: Low costs
   - Cons: Significant drift possible
   
4. **Threshold-based**: Rebalance when drift > X%
   - Pros: Best of both worlds
   - Cons: Irregular schedule

**Recommendations by strategy**:
| Strategy Type | Rebalance Frequency | Threshold |
|--------------|-------------------|-----------|
| Day trading | Daily | 10% |
| Swing trading | Weekly | 15% |
| Position trading | Monthly | 20% |
| Long-term hold | Quarterly | 25% |

### What's diversification ratio?

**Formula**:
```
DR = (Weighted avg of individual volatilities) / (Portfolio volatility)
DR = (Σ w_i * σ_i) / σ_portfolio
```

**Interpretation**:
- DR = 1: No diversification benefit (perfect correlation)
- DR = 2: Portfolio vol is half what you'd expect from sum of parts
- DR = √N: Perfect diversification (N assets, zero correlation)

**Example**:
- 4 assets, equal weight, σ_i = 0.04 each
- If perfectly correlated: σ_portfolio = 0.04, DR = 1
- If uncorrelated: σ_portfolio = 0.02, DR = 2
- Maximum possible DR = √4 = 2

**Use case**: Assess portfolio construction quality. Higher DR = better diversification.

### What's the correlation matrix telling me?

**Matrix structure**:
```
         BTC   ETH   BNB
BTC     1.00  0.85  0.70
ETH     0.85  1.00  0.75
BNB     0.70  0.75  1.00
```

**Interpretation**:
- **1.0**: Perfect positive correlation (always move together)
- **0.0**: No correlation (independent)
- **-1.0**: Perfect negative correlation (always move opposite)

**For crypto**:
- BTC/ETH typically 0.7-0.9 (high correlation)
- BTC/stablecoin ~0.0 (no correlation)
- Long BTC/Short BTC = -1.0 (perfect hedge)

**Diversification implications**:
- Correlations > 0.8: Limited diversification benefit
- Correlations 0.3-0.7: Good diversification
- Correlations < 0.3: Excellent diversification

**Watch for**: Correlation can change during market stress (tends toward 1.0 in crashes).

---

## Performance & Troubleshooting

### Why is the UI slow?

**Common causes**:

1. **Large dataset**:
   - Years of 30m data
   - Solution: Use higher timeframe or shorter date range

2. **Complex visualizations**:
   - Welch heatmap with high resolution
   - Solution: Reduce window length or disable heatmap

3. **Optimization running**:
   - Bayesian search with many iterations
   - Solution: Use random search or reduce iterations

4. **No caching**:
   - Force refresh enabled
   - Solution: Disable force refresh for repeated loads

**Performance tips**:
- Use 1h or 4h data (not 30m) for backtests
- Run optimizations in batches
- Close unused visualizations
- Reduce spectral analysis window sizes

### Can I run backtests in parallel?

**Portfolio backtests**: Yes, automatically parallelized.
- Uses `ProcessPoolExecutor`
- Scales to N_cores

**Single-symbol optimizations**: Yes.
- Grid/Random search: Parallel by default
- Bayesian: Sequential by design (each iteration depends on previous)

**To control parallelism**:
```python
# In portfolio/executor.py
n_workers = min(cpu_count(), len(symbols))
```

### How much memory does it use?

**Estimates**:
| Task | Memory Usage |
|------|-------------|
| Load 1 year of 1h data | ~50 MB |
| Basic backtest | ~100 MB |
| FFT/Welch analysis | +50 MB |
| Grid search (100 combos) | ~500 MB |
| Portfolio (5 symbols) | ~200 MB |

**For large tasks**:
- 10 symbols, 2 years, optimization: ~2-4 GB
- Reduce parallelism if OOM errors occur

### Why don't my trades match the chart signals?

**Common reasons**:

1. **Next-bar fill**:
   - Signal at bar 100 close → Fill at bar 101 open
   - Visual: Marker on bar 100, but trade entry is bar 101

2. **Stop loss hit**:
   - Entry signal generated but stop hit before next bar
   - Trade exited immediately

3. **Position already open**:
   - New signal ignored if already in position
   - Single position enforced

4. **Insufficient capital**:
   - Signal generated but not enough capital after fees
   - Trade skipped

**Debug steps**:
1. Enable "Show Trade Log"
2. Check entry/exit prices vs. signals
3. Verify stop levels on chart
4. Check equity curve for capital availability

### How do I debug a strategy?

**Step-by-step**:

1. **Visual inspection**:
   - Plot price + smoothed trend
   - Check if trend makes sense
   - Verify signal markers align with strategy logic

2. **Trade log analysis**:
   - Export trades to CSV
   - Check entry/exit conditions
   - Verify P&L calculations

3. **Metric validation**:
   - Compute metrics manually on sample trades
   - Ensure consistency with reported values

4. **Incremental testing**:
   - Start with simple strategy (no stops)
   - Add complexity one piece at a time
   - Isolate where results diverge

5. **Parameter sensitivity**:
   - Vary one parameter at a time
   - Check if results change as expected
   - Ensure no unexpected discontinuities

**Tools**:
- Streamlit UI for visual debugging
- Jupyter notebook for detailed analysis
- Python debugger (pdb) for code-level inspection

---

## Advanced Topics

### Can I use this for live trading?

**Current state**: No live execution built-in.

**Options**:

1. **Manual trading**:
   - Run backtest daily
   - Manually place trades on exchange
   - Use trade log as guide

2. **Integration via API**:
   - Use backtest engine for signals
   - Implement execution layer separately
   - Connect to Binance trading API
   - **Caution**: Handle errors, slippage, partial fills

3. **Paper trading**:
   - Use Binance testnet
   - Simulate live execution
   - Validate without real capital

**Recommendation**: Master backtesting and optimization first, then consider live deployment with small capital.

### How do I add a new strategy?

**Steps**:

1. **Define signal logic** in `core/analysis/signals.py`:
   ```python
   def generate_my_strategy_signals(
       close: np.ndarray,
       indicator: np.ndarray,
       threshold: float
   ) -> np.ndarray:
       # Your logic here
       return signals
   ```

2. **Integrate with backtest**:
   ```python
   from core.analysis.signals import generate_my_strategy_signals
   signals = generate_my_strategy_signals(close, indicator, threshold)
   result = run_backtest(signals, ...)
   ```

3. **Add to UI** (optional):
   - Edit `app/ui/main.py`
   - Add parameter controls
   - Connect to backtest button

4. **Test thoroughly**:
   - Write unit tests
   - Validate on multiple symbols/periods
   - Check for edge cases

### Can I combine multiple signals?

**Yes**, several approaches:

1. **Logical AND** (require all):
   ```python
   combined_signals = signal_1 & signal_2 & signal_3
   ```

2. **Logical OR** (any):
   ```python
   combined_signals = signal_1 | signal_2 | signal_3
   ```

3. **Weighted voting**:
   ```python
   score = 0.5 * signal_1 + 0.3 * signal_2 + 0.2 * signal_3
   combined_signals = score > threshold
   ```

4. **Sequential filtering**:
   ```python
   # Start with all entries
   entries = trend_signal
   # Filter by MTF
   entries = entries & mtf_aligned
   # Filter by volatility
   entries = entries & (volatility > min_vol)
   ```

### Can I use this for other asset classes?

**In principle**: Yes, if you provide OHLCV data.

**Tested on**: Crypto only.

**Considerations for other assets**:

1. **Stocks**:
   - Market hours (gaps at open/close)
   - Dividends (adjust for splits/dividends)
   - Lower volatility (adjust parameters)

2. **Forex**:
   - 24/5 trading (weekend gaps)
   - Swap fees (overnight interest)
   - Lower volatility (adjust stops)

3. **Commodities**:
   - Futures contracts (roll costs)
   - Seasonality
   - Backwardation/contango

**To implement**:
- Create new data loader for your data source
- Adjust default parameters for asset volatility
- Test thoroughly before relying on results

---

## Getting Help

### Where can I ask questions?

1. **GitHub Issues**: Bug reports and feature requests
2. **GitHub Discussions**: General questions and strategy discussions
3. **Documentation**: Check [README.md](../README.md), [ARCHITECTURE.md](ARCHITECTURE.md), [CONFIGURATION.md](CONFIGURATION.md)

### How do I report a bug?

Include:
1. **Error message** (full traceback)
2. **Steps to reproduce**
3. **Configuration** (export preset)
4. **Data** (symbol, interval, date range)
5. **Environment** (Python version, OS)

### How can I contribute?

See [Contributing section in README](../README.md#contributing).

**Areas welcome**:
- New signal generators
- Additional exchanges
- Performance optimizations
- Documentation improvements
- Test coverage

---

**Last updated**: 2024

For more information, see:
- [README.md](../README.md) - Main documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical deep dive
- [CONFIGURATION.md](CONFIGURATION.md) - Parameter reference
