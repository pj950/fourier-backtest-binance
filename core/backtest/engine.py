from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    initial_capital: float = 10000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    position_size_mode: str = "full"
    position_size_fraction: float = 1.0


@dataclass
class Trade:
    """Record of a single trade."""

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


@dataclass
class BacktestResult:
    """Complete backtesting results."""

    equity_curve: np.ndarray
    trades: list[Trade]
    metrics: dict[str, float]


def compute_mae_mfe(
    entry_price: float,
    prices_high: np.ndarray,
    prices_low: np.ndarray,
    entry_idx: int,
    exit_idx: int,
    direction: int,
) -> tuple[float, float]:
    """
    Compute Maximum Adverse Excursion and Maximum Favorable Excursion.

    Args:
        entry_price: Entry price
        prices_high: High prices
        prices_low: Low prices
        entry_idx: Entry bar index
        exit_idx: Exit bar index
        direction: 1 for long, -1 for short

    Returns:
        Tuple of (MAE, MFE)
    """
    if entry_idx >= exit_idx or exit_idx > len(prices_high):
        return 0.0, 0.0

    window_high = prices_high[entry_idx:exit_idx]
    window_low = prices_low[entry_idx:exit_idx]

    if direction == 1:
        mfe = np.max(window_high) - entry_price
        mae = entry_price - np.min(window_low)
    else:
        mfe = entry_price - np.min(window_low)
        mae = np.max(window_high) - entry_price

    return mae, mfe


def run_backtest(
    signals: np.ndarray,
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    timestamps: pd.DatetimeIndex,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Run vectorized backtest with next-bar open fills.

    Signals:
    - 1: Enter long at next bar open
    - -1: Exit position at next bar open
    - 0: Hold current position

    Args:
        signals: Signal array (1=entry, -1=exit, 0=hold)
        open_prices: Open prices
        high_prices: High prices
        low_prices: Low prices
        close_prices: Close prices
        timestamps: Timestamps for each bar
        config: Backtest configuration

    Returns:
        BacktestResult with equity curve, trades, and metrics
    """
    if config is None:
        config = BacktestConfig()

    n = len(signals)
    equity = np.full(n, config.initial_capital)
    cash = config.initial_capital
    position = 0.0
    position_price = 0.0
    position_entry_idx = -1

    trades: list[Trade] = []

    for i in range(n - 1):
        current_signal = signals[i]
        next_open = open_prices[i + 1]

        equity[i] = cash + position * close_prices[i]

        if current_signal == 1 and position == 0:
            fill_price = next_open * (1 + config.slippage)

            if config.position_size_mode == "full":
                max_spending = cash * config.position_size_fraction
                size = max_spending / (fill_price * (1 + config.fee_rate))
                cost = size * fill_price
                fees = cost * config.fee_rate
                total_cost = cost + fees
            else:
                size = config.position_size_fraction
                cost = size * fill_price
                fees = cost * config.fee_rate
                total_cost = cost + fees

            if total_cost <= cash + 1e-10:
                position = size
                position_price = fill_price
                position_entry_idx = i + 1
                cash -= total_cost

        elif current_signal == -1 and position > 0:
            fill_price = next_open * (1 - config.slippage)

            proceeds = position * fill_price
            fees = proceeds * config.fee_rate
            net_proceeds = proceeds - fees

            cash += net_proceeds

            pnl = net_proceeds - (position * position_price + position * position_price * config.fee_rate)
            pnl_pct = pnl / (position * position_price)

            mae, mfe = compute_mae_mfe(
                entry_price=position_price,
                prices_high=high_prices,
                prices_low=low_prices,
                entry_idx=position_entry_idx,
                exit_idx=i + 1,
                direction=1,
            )

            mae_pct = mae / position_price if position_price > 0 else 0.0
            mfe_pct = mfe / position_price if position_price > 0 else 0.0

            bars_held = (i + 1) - position_entry_idx

            trade = Trade(
                entry_idx=position_entry_idx,
                exit_idx=i + 1,
                entry_time=timestamps[position_entry_idx],
                exit_time=timestamps[i + 1],
                entry_price=position_price,
                exit_price=fill_price,
                size=position,
                pnl=pnl,
                pnl_pct=pnl_pct,
                mae=mae,
                mfe=mfe,
                mae_pct=mae_pct,
                mfe_pct=mfe_pct,
                bars_held=bars_held,
                fees=fees + position * position_price * config.fee_rate,
            )
            trades.append(trade)

            position = 0.0
            position_price = 0.0
            position_entry_idx = -1

    equity[n - 1] = cash + position * close_prices[n - 1]

    metrics = compute_metrics(equity, trades, config.initial_capital)

    return BacktestResult(equity_curve=equity, trades=trades, metrics=metrics)


def compute_metrics(
    equity_curve: np.ndarray,
    trades: list[Trade],
    initial_capital: float,
) -> dict[str, float]:
    """
    Compute performance metrics from equity curve and trades.

    Args:
        equity_curve: Equity values over time
        trades: List of completed trades
        initial_capital: Starting capital

    Returns:
        Dictionary of metrics
    """
    if len(equity_curve) == 0:
        return {}

    total_return = (equity_curve[-1] - initial_capital) / initial_capital
    cum_return = equity_curve[-1] / initial_capital - 1.0

    n_bars = len(equity_curve)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[~np.isnan(returns)]

    max_dd, max_dd_pct = compute_max_drawdown(equity_curve)

    sharpe = compute_sharpe_ratio(returns, periods_per_year=365 * 24)
    sortino = compute_sortino_ratio(returns, periods_per_year=365 * 24)

    n_trades = len(trades)
    if n_trades > 0:
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        n_wins = len(winning_trades)
        n_losses = len(losing_trades)

        win_rate = n_wins / n_trades if n_trades > 0 else 0.0

        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))

        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

        avg_win = total_profit / n_wins if n_wins > 0 else 0.0
        avg_loss = total_loss / n_losses if n_losses > 0 else 0.0

        avg_bars_held = np.mean([t.bars_held for t in trades])
        avg_mae = np.mean([t.mae for t in trades])
        avg_mfe = np.mean([t.mfe for t in trades])
        avg_mae_pct = np.mean([t.mae_pct for t in trades])
        avg_mfe_pct = np.mean([t.mfe_pct for t in trades])
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        avg_bars_held = 0.0
        avg_mae = 0.0
        avg_mfe = 0.0
        avg_mae_pct = 0.0
        avg_mfe_pct = 0.0
        n_wins = 0
        n_losses = 0

    periods_per_year = 365 * 24
    annualized_return = (1 + cum_return) ** (periods_per_year / n_bars) - 1 if n_bars > 0 else 0.0

    return {
        "total_return": total_return,
        "cumulative_return": cum_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "n_trades": n_trades,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_bars_held": avg_bars_held,
        "avg_mae": avg_mae,
        "avg_mfe": avg_mfe,
        "avg_mae_pct": avg_mae_pct,
        "avg_mfe_pct": avg_mfe_pct,
    }


def compute_max_drawdown(equity_curve: np.ndarray) -> tuple[float, float]:
    """
    Compute maximum drawdown in absolute and percentage terms.

    Args:
        equity_curve: Equity values over time

    Returns:
        Tuple of (max_drawdown_absolute, max_drawdown_percentage)
    """
    if len(equity_curve) == 0:
        return 0.0, 0.0

    cummax = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - cummax
    max_dd = np.min(drawdown)

    max_dd_pct = max_dd / cummax[np.argmin(drawdown)] if cummax[np.argmin(drawdown)] > 0 else 0.0

    return max_dd, max_dd_pct


def compute_sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Compute annualized Sharpe ratio.

    Args:
        returns: Array of period returns
        periods_per_year: Number of periods in a year for annualization

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    if std_return < 1e-10:
        return 0.0

    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    return sharpe


def compute_sortino_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
    target_return: float = 0.0,
) -> float:
    """
    Compute annualized Sortino ratio.

    Args:
        returns: Array of period returns
        periods_per_year: Number of periods in a year for annualization
        target_return: Target return (default 0 for downside deviation)

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    mean_return = np.mean(returns)
    downside_returns = returns[returns < target_return]

    if len(downside_returns) == 0:
        return np.inf

    downside_std = np.std(downside_returns, ddof=1)

    if downside_std == 0:
        return np.inf

    sortino = (mean_return - target_return) / downside_std * np.sqrt(periods_per_year)
    return sortino


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    """
    Convert list of trades to pandas DataFrame.

    Args:
        trades: List of Trade objects

    Returns:
        DataFrame with trade information
    """
    if not trades:
        return pd.DataFrame()

    return pd.DataFrame([vars(t) for t in trades])
