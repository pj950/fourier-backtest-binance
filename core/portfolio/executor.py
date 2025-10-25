from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from core.backtest.engine import BacktestConfig, BacktestResult, run_backtest


@dataclass
class SymbolBacktestResult:
    """Results for a single symbol backtest."""

    symbol: str
    result: BacktestResult
    returns: np.ndarray
    equity_curve: np.ndarray


def run_single_symbol_backtest(
    symbol: str,
    df: pd.DataFrame,
    strategy_func: Callable,
    strategy_params: dict,
    config: BacktestConfig,
) -> SymbolBacktestResult:
    """
    Run backtest for a single symbol.

    Args:
        symbol: Trading symbol
        df: OHLCV DataFrame for the symbol
        strategy_func: Function that generates signals from df and params
        strategy_params: Parameters for strategy function
        config: Backtest configuration

    Returns:
        SymbolBacktestResult with backtest results and returns
    """
    # Generate signals using strategy function
    signals = strategy_func(df, **strategy_params)

    # Extract price data
    open_prices = df["open"].values
    high_prices = df["high"].values
    low_prices = df["low"].values
    close_prices = df["close"].values
    timestamps = df["open_time"]

    # Run backtest
    result = run_backtest(
        signals=signals,
        open_prices=open_prices,
        high_prices=high_prices,
        low_prices=low_prices,
        close_prices=close_prices,
        timestamps=timestamps,
        config=config,
    )

    # Compute returns
    returns = np.diff(result.equity_curve) / result.equity_curve[:-1]
    returns = np.concatenate([[0.0], returns])  # Prepend 0 for alignment

    return SymbolBacktestResult(
        symbol=symbol,
        result=result,
        returns=returns,
        equity_curve=result.equity_curve,
    )


def run_parallel_backtests(
    symbols: list[str],
    data_dict: dict[str, pd.DataFrame],
    strategy_func: Callable,
    strategy_params: dict,
    config: BacktestConfig,
    max_workers: int | None = None,
) -> dict[str, SymbolBacktestResult]:
    """
    Run backtests for multiple symbols in parallel.

    Args:
        symbols: List of trading symbols
        data_dict: Dictionary mapping symbol to OHLCV DataFrame
        strategy_func: Function that generates signals from df and params
        strategy_params: Parameters for strategy function
        config: Backtest configuration
        max_workers: Maximum number of parallel workers (None = CPU count)

    Returns:
        Dictionary mapping symbol to SymbolBacktestResult
    """
    results = {}

    # For now, run sequentially to avoid pickling issues with strategy_func
    # TODO: Implement proper parallel execution with serializable functions
    for symbol in symbols:
        if symbol not in data_dict:
            continue

        df = data_dict[symbol]
        result = run_single_symbol_backtest(
            symbol=symbol,
            df=df,
            strategy_func=strategy_func,
            strategy_params=strategy_params,
            config=config,
        )
        results[symbol] = result

    return results


def align_symbol_results(
    results: dict[str, SymbolBacktestResult],
    method: str = "intersection",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align equity curves and returns across symbols to common timeline.

    Args:
        results: Dictionary of symbol results
        method: Alignment method ('intersection' or 'union')

    Returns:
        Tuple of (equity_df, returns_df) with aligned data
    """
    if not results:
        return pd.DataFrame(), pd.DataFrame()

    # Build DataFrames for equity curves and returns
    equity_data = {}
    returns_data = {}

    # Get timestamps from first symbol to use as reference
    first_symbol = list(results.keys())[0]
    first_result = results[first_symbol]
    
    # For simplicity, we'll assume all symbols have the same length
    # In production, you'd want to properly align by timestamp
    for symbol, symbol_result in results.items():
        equity_data[symbol] = symbol_result.equity_curve
        returns_data[symbol] = symbol_result.returns

    equity_df = pd.DataFrame(equity_data)
    returns_df = pd.DataFrame(returns_data)

    return equity_df, returns_df


def compute_symbol_statistics(
    results: dict[str, SymbolBacktestResult],
) -> pd.DataFrame:
    """
    Compute summary statistics for each symbol.

    Args:
        results: Dictionary of symbol results

    Returns:
        DataFrame with statistics for each symbol
    """
    stats = []

    for symbol, result in results.items():
        metrics = result.result.metrics
        stats.append({
            "symbol": symbol,
            "total_return": metrics.get("total_return", 0.0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
            "max_drawdown_pct": metrics.get("max_drawdown_pct", 0.0),
            "n_trades": metrics.get("n_trades", 0),
            "win_rate": metrics.get("win_rate", 0.0),
            "profit_factor": metrics.get("profit_factor", 0.0),
        })

    return pd.DataFrame(stats)
