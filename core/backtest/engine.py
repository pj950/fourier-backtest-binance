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
    allow_shorts: bool = False
    max_bars_held: int | None = None
    enable_partial_tp: bool = False
    partial_tp_scales: list[tuple[float, float]] | None = None
    enable_pyramiding: bool = False
    max_pyramids: int = 3
    pyramid_scale: float = 0.5
    sizing_mode: str = "fixed"
    volatility_target: float = 0.02
    max_risk_per_trade: float = 0.02


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
    direction: int = 1
    exit_reason: str = "signal"
    partial_exits: list[tuple[int, float, float]] | None = None


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


def run_backtest_enhanced(
    signals: np.ndarray,
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    timestamps: pd.DatetimeIndex,
    atr: np.ndarray | None = None,
    stop_levels: np.ndarray | None = None,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Run enhanced backtest with support for shorts, partial exits, time stops, and dynamic sizing.

    Signals:
    - 1: Enter long at next bar open
    - -1: Exit position at next bar open (or enter short if allow_shorts=True)
    - 0: Hold current position
    - 2: Enter short at next bar open (if allow_shorts=True)
    - -2: Exit short position

    Args:
        signals: Signal array
        open_prices: Open prices
        high_prices: High prices
        low_prices: Low prices
        close_prices: Close prices
        timestamps: Timestamps for each bar
        atr: ATR values for dynamic sizing (optional)
        stop_levels: Stop loss levels (optional)
        config: Backtest configuration

    Returns:
        BacktestResult with equity curve, trades, and metrics
    """
    if config is None:
        config = BacktestConfig()

    from core.analysis.exits import check_time_based_exit, compute_partial_tp_levels, check_partial_tp_hit
    from core.analysis.sizing import compute_volatility_based_size, compute_fixed_risk_size

    n = len(signals)
    equity = np.full(n, config.initial_capital)
    cash = config.initial_capital
    position = 0.0
    position_price = 0.0
    position_entry_idx = -1
    position_direction = 0

    partial_tp_levels: list[tuple[float, float]] = []
    hit_tp_levels: set[int] = set()

    trades: list[Trade] = []

    for i in range(n - 1):
        current_signal = signals[i]
        next_open = open_prices[i + 1]

        equity[i] = cash + abs(position) * close_prices[i]

        in_position = abs(position) > 1e-10

        if in_position:
            exit_triggered = False
            exit_reason = "signal"

            if config.max_bars_held is not None:
                if check_time_based_exit(position_entry_idx, i, config.max_bars_held):
                    exit_triggered = True
                    exit_reason = "time"

            if stop_levels is not None and len(stop_levels) > i:
                if position_direction == 1 and close_prices[i] < stop_levels[i]:
                    exit_triggered = True
                    exit_reason = "stop"
                elif position_direction == -1 and close_prices[i] > stop_levels[i]:
                    exit_triggered = True
                    exit_reason = "stop"

            if config.enable_partial_tp and partial_tp_levels:
                newly_hit = check_partial_tp_hit(
                    close_prices[i],
                    high_prices[i],
                    low_prices[i],
                    partial_tp_levels,
                    hit_tp_levels,
                    position_direction,
                )
                for level_idx in newly_hit:
                    hit_tp_levels.add(level_idx)

            signal_exit = (
                (position_direction == 1 and current_signal == -1) or
                (position_direction == -1 and current_signal == -2)
            )

            if signal_exit or exit_triggered:
                fill_price = next_open * (1 - config.slippage) if position_direction == 1 else next_open * (1 + config.slippage)

                proceeds = abs(position) * fill_price
                fees = proceeds * config.fee_rate
                net_proceeds = proceeds - fees

                if position_direction == 1:
                    cash += net_proceeds
                    pnl = net_proceeds - (abs(position) * position_price + abs(position) * position_price * config.fee_rate)
                else:
                    entry_cost = abs(position) * position_price
                    exit_cost = abs(position) * fill_price
                    pnl = entry_cost - exit_cost - fees - entry_cost * config.fee_rate
                    cash += entry_cost + pnl

                pnl_pct = pnl / (abs(position) * position_price)

                mae, mfe = compute_mae_mfe(
                    entry_price=position_price,
                    prices_high=high_prices,
                    prices_low=low_prices,
                    entry_idx=position_entry_idx,
                    exit_idx=i + 1,
                    direction=position_direction,
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
                    size=abs(position),
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    mae=mae,
                    mfe=mfe,
                    mae_pct=mae_pct,
                    mfe_pct=mfe_pct,
                    bars_held=bars_held,
                    fees=fees + abs(position) * position_price * config.fee_rate,
                    direction=position_direction,
                    exit_reason=exit_reason,
                )
                trades.append(trade)

                position = 0.0
                position_price = 0.0
                position_entry_idx = -1
                position_direction = 0
                partial_tp_levels = []
                hit_tp_levels = set()

        if not in_position:
            is_long_entry = current_signal == 1
            is_short_entry = config.allow_shorts and current_signal == 2

            if is_long_entry or is_short_entry:
                direction = 1 if is_long_entry else -1
                fill_price = next_open * (1 + config.slippage) if direction == 1 else next_open * (1 - config.slippage)

                if config.sizing_mode == "volatility" and atr is not None and stop_levels is not None:
                    current_atr = atr[i] if i < len(atr) else atr[-1]
                    stop_price = stop_levels[i] if i < len(stop_levels) else fill_price * 0.95

                    size = compute_volatility_based_size(
                        capital=cash,
                        entry_price=fill_price,
                        stop_price=stop_price,
                        volatility=current_atr,
                        risk_target=config.volatility_target,
                        max_risk_per_trade=config.max_risk_per_trade,
                    )
                elif config.sizing_mode == "fixed_risk" and stop_levels is not None:
                    stop_price = stop_levels[i] if i < len(stop_levels) else fill_price * 0.95
                    size = compute_fixed_risk_size(
                        capital=cash,
                        entry_price=fill_price,
                        stop_price=stop_price,
                        risk_fraction=config.max_risk_per_trade,
                    )
                else:
                    if config.position_size_mode == "full":
                        max_spending = cash * config.position_size_fraction
                        size = max_spending / (fill_price * (1 + config.fee_rate))
                    else:
                        size = config.position_size_fraction

                cost = size * fill_price
                fees = cost * config.fee_rate
                total_cost = cost + fees

                if total_cost <= cash + 1e-10 and size > 0:
                    position = size if direction == 1 else -size
                    position_price = fill_price
                    position_entry_idx = i + 1
                    position_direction = direction
                    cash -= total_cost

                    if config.enable_partial_tp and config.partial_tp_scales:
                        from core.analysis.exits import compute_partial_tp_levels as compute_levels
                        partial_tp_levels = compute_levels(
                            fill_price,
                            direction,
                            config.partial_tp_scales,
                        )
                        hit_tp_levels = set()

    equity[n - 1] = cash + abs(position) * close_prices[n - 1]

    metrics = compute_metrics(equity, trades, config.initial_capital)

    return BacktestResult(equity_curve=equity, trades=trades, metrics=metrics)


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
