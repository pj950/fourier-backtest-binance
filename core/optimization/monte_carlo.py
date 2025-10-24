"""Monte Carlo resampling and bootstrap analysis."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo analysis."""

    n_simulations: int
    metrics_distributions: dict[str, np.ndarray]
    percentiles: dict[str, dict[str, float]]
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]
    equity_curves: list[np.ndarray] | None = None


def block_bootstrap_returns(
    returns: np.ndarray,
    n_simulations: int = 1000,
    block_size: int = 20,
    seed: int | None = None,
) -> list[np.ndarray]:
    """
    Generate bootstrap samples using block bootstrap.

    Args:
        returns: Original return series
        n_simulations: Number of bootstrap samples
        block_size: Block size for bootstrap
        seed: Random seed

    Returns:
        List of resampled return arrays
    """
    rng = np.random.RandomState(seed)
    n_returns = len(returns)

    if block_size <= 0:
        block_size = int(np.sqrt(n_returns))

    n_blocks = int(np.ceil(n_returns / block_size))

    bootstrap_samples = []

    for _ in range(n_simulations):
        resampled = []

        for _ in range(n_blocks):
            start_idx = rng.randint(0, max(1, n_returns - block_size + 1))
            end_idx = min(start_idx + block_size, n_returns)
            block = returns[start_idx:end_idx]
            resampled.extend(block)

        resampled = np.array(resampled[:n_returns])
        bootstrap_samples.append(resampled)

    return bootstrap_samples


def block_bootstrap_residuals(
    prices: np.ndarray,
    smoothed: np.ndarray,
    n_simulations: int = 1000,
    block_size: int = 20,
    seed: int | None = None,
) -> list[np.ndarray]:
    """
    Generate bootstrap samples by resampling residuals.

    Args:
        prices: Original price series
        smoothed: Smoothed/fitted price series
        n_simulations: Number of bootstrap samples
        block_size: Block size for bootstrap
        seed: Random seed

    Returns:
        List of synthetic price arrays
    """
    residuals = prices - smoothed
    residuals = residuals[~np.isnan(residuals)]

    bootstrap_residuals = block_bootstrap_returns(residuals, n_simulations, block_size, seed)

    bootstrap_prices = []
    for res_sample in bootstrap_residuals:
        synthetic_prices = smoothed[:len(res_sample)] + res_sample
        bootstrap_prices.append(synthetic_prices)

    return bootstrap_prices


def monte_carlo_equity_curves(
    equity_curve: np.ndarray,
    n_simulations: int = 1000,
    block_size: int = 20,
    seed: int | None = None,
) -> list[np.ndarray]:
    """
    Generate Monte Carlo equity curves by bootstrapping returns.

    Args:
        equity_curve: Original equity curve
        n_simulations: Number of simulations
        block_size: Block size for bootstrap
        seed: Random seed

    Returns:
        List of simulated equity curves
    """
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[~np.isnan(returns)]

    bootstrap_returns = block_bootstrap_returns(returns, n_simulations, block_size, seed)

    simulated_curves = []
    initial_equity = equity_curve[0]

    for ret_sample in bootstrap_returns:
        sim_equity = np.zeros(len(ret_sample) + 1)
        sim_equity[0] = initial_equity

        for i, r in enumerate(ret_sample):
            sim_equity[i + 1] = sim_equity[i] * (1 + r)

        simulated_curves.append(sim_equity)

    return simulated_curves


def compute_mc_metrics(equity_curves: list[np.ndarray], initial_capital: float = 10000.0) -> MonteCarloResult:
    """
    Compute metrics distributions from Monte Carlo equity curves.

    Args:
        equity_curves: List of simulated equity curves
        initial_capital: Initial capital

    Returns:
        MonteCarloResult with metric distributions
    """
    n_simulations = len(equity_curves)

    metrics_distributions = {
        "total_return": np.zeros(n_simulations),
        "sharpe_ratio": np.zeros(n_simulations),
        "sortino_ratio": np.zeros(n_simulations),
        "max_drawdown_pct": np.zeros(n_simulations),
        "annualized_return": np.zeros(n_simulations),
    }

    for i, equity in enumerate(equity_curves):
        if len(equity) < 2:
            continue

        total_return = (equity[-1] - initial_capital) / initial_capital
        metrics_distributions["total_return"][i] = total_return

        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns)]

        if len(returns) > 1:
            sharpe = np.mean(returns) / (np.std(returns, ddof=1) + 1e-10) * np.sqrt(365 * 24)
            metrics_distributions["sharpe_ratio"][i] = sharpe

            downside = returns[returns < 0]
            if len(downside) > 0:
                sortino = np.mean(returns) / (np.std(downside, ddof=1) + 1e-10) * np.sqrt(365 * 24)
            else:
                sortino = np.inf
            metrics_distributions["sortino_ratio"][i] = sortino

        cummax = np.maximum.accumulate(equity)
        drawdown = equity - cummax
        max_dd = np.min(drawdown)
        max_dd_pct = max_dd / cummax[np.argmin(drawdown)] if cummax[np.argmin(drawdown)] > 0 else 0.0
        metrics_distributions["max_drawdown_pct"][i] = max_dd_pct

        n_bars = len(equity)
        cum_return = equity[-1] / initial_capital - 1.0
        annualized = (1 + cum_return) ** (365 * 24 / n_bars) - 1 if n_bars > 0 else 0.0
        metrics_distributions["annualized_return"][i] = annualized

    percentiles = {}
    mean_metrics = {}
    std_metrics = {}

    for metric_name, values in metrics_distributions.items():
        values = values[~np.isinf(values)]
        percentiles[metric_name] = {
            "p5": float(np.percentile(values, 5)),
            "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
        }
        mean_metrics[metric_name] = float(np.mean(values))
        std_metrics[metric_name] = float(np.std(values, ddof=1))

    return MonteCarloResult(
        n_simulations=n_simulations,
        metrics_distributions=metrics_distributions,
        percentiles=percentiles,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        equity_curves=equity_curves,
    )
