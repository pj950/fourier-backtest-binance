"""Walk-forward analysis for parameter optimization."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WalkForwardWindow:
    """A single walk-forward window with train/test split."""

    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    window_id: int


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis."""

    windows: list[WalkForwardWindow]
    train_metrics: list[dict[str, float]]
    test_metrics: list[dict[str, float]]
    best_params_per_window: list[dict]
    oos_equity_curves: list[np.ndarray]
    combined_oos_metrics: dict[str, float]


def create_walkforward_windows(
    n_bars: int,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    anchored: bool = False,
) -> list[WalkForwardWindow]:
    """
    Create walk-forward windows with train/test splits.

    Args:
        n_bars: Total number of bars
        train_size: Number of bars in training window
        test_size: Number of bars in test window
        step_size: Step size for rolling window (defaults to test_size)
        anchored: If True, training window grows (anchored walk-forward)

    Returns:
        List of WalkForwardWindow objects
    """
    if step_size is None:
        step_size = test_size

    windows = []
    window_id = 0

    if anchored:
        train_start = 0
        current_pos = train_size

        while current_pos + test_size <= n_bars:
            train_end = current_pos
            test_start = current_pos
            test_end = current_pos + test_size

            windows.append(
                WalkForwardWindow(
                    train_start_idx=train_start,
                    train_end_idx=train_end,
                    test_start_idx=test_start,
                    test_end_idx=test_end,
                    window_id=window_id,
                )
            )

            current_pos += step_size
            window_id += 1
    else:
        current_pos = 0

        while current_pos + train_size + test_size <= n_bars:
            train_start = current_pos
            train_end = current_pos + train_size
            test_start = train_end
            test_end = test_start + test_size

            windows.append(
                WalkForwardWindow(
                    train_start_idx=train_start,
                    train_end_idx=train_end,
                    test_start_idx=test_start,
                    test_end_idx=test_end,
                    window_id=window_id,
                )
            )

            current_pos += step_size
            window_id += 1

    return windows


def split_data_by_window(
    df: pd.DataFrame,
    window: WalkForwardWindow,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe by walk-forward window.

    Args:
        df: Input dataframe
        window: Walk-forward window

    Returns:
        Tuple of (train_df, test_df)
    """
    train_df = df.iloc[window.train_start_idx : window.train_end_idx].copy()
    test_df = df.iloc[window.test_start_idx : window.test_end_idx].copy()

    return train_df, test_df


def compute_combined_oos_metrics(equity_curves: list[np.ndarray]) -> dict[str, float]:
    """
    Compute combined out-of-sample metrics from multiple equity curves.

    Args:
        equity_curves: List of equity curves from each test window

    Returns:
        Dictionary of combined metrics
    """
    if not equity_curves:
        return {}

    combined_equity = np.concatenate(equity_curves)

    if len(combined_equity) == 0:
        return {}

    initial_capital = combined_equity[0] if len(combined_equity) > 0 else 10000.0
    final_capital = combined_equity[-1]

    total_return = (final_capital - initial_capital) / initial_capital
    cum_return = final_capital / initial_capital - 1.0

    returns = np.diff(combined_equity) / combined_equity[:-1]
    returns = returns[~np.isnan(returns)]

    if len(returns) > 0:
        sharpe = np.mean(returns) / (np.std(returns, ddof=1) + 1e-10) * np.sqrt(365 * 24)

        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            sortino = np.mean(returns) / (np.std(downside_returns, ddof=1) + 1e-10) * np.sqrt(365 * 24)
        else:
            sortino = np.inf
    else:
        sharpe = 0.0
        sortino = 0.0

    cummax = np.maximum.accumulate(combined_equity)
    drawdown = combined_equity - cummax
    max_dd = np.min(drawdown)
    max_dd_pct = max_dd / cummax[np.argmin(drawdown)] if cummax[np.argmin(drawdown)] > 0 else 0.0

    n_bars = len(combined_equity)
    periods_per_year = 365 * 24
    annualized_return = (1 + cum_return) ** (periods_per_year / n_bars) - 1 if n_bars > 0 else 0.0

    return {
        "total_return": total_return,
        "cumulative_return": cum_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "n_windows": len(equity_curves),
    }
