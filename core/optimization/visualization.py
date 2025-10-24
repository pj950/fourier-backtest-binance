"""Visualization utilities for optimization results."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from core.optimization.monte_carlo import MonteCarloResult
from core.optimization.runner import OptimizationRun
from core.optimization.walkforward import WalkForwardResult


def plot_param_heatmap(
    leaderboard: pd.DataFrame,
    param_x: str,
    param_y: str,
    metric: str = "train_sharpe_ratio",
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot heatmap of metric values across two parameters.

    Args:
        leaderboard: Leaderboard dataframe
        param_x: Parameter for x-axis
        param_y: Parameter for y-axis
        metric: Metric to visualize
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if param_x not in leaderboard.columns or param_y not in leaderboard.columns:
        raise ValueError(f"Parameters {param_x} or {param_y} not in leaderboard")

    if metric not in leaderboard.columns:
        raise ValueError(f"Metric {metric} not in leaderboard")

    pivot_data = leaderboard.pivot_table(
        index=param_y, columns=param_x, values=metric, aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax, cbar_kws={"label": metric})
    ax.set_title(f"{metric} Heatmap")
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)

    return fig


def plot_frontier(
    leaderboard: pd.DataFrame,
    metric_x: str = "train_sharpe_ratio",
    metric_y: str = "train_max_drawdown_pct",
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot efficient frontier of two metrics.

    Args:
        leaderboard: Leaderboard dataframe
        metric_x: Metric for x-axis (e.g., return)
        metric_y: Metric for y-axis (e.g., risk)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if metric_x not in leaderboard.columns or metric_y not in leaderboard.columns:
        raise ValueError(f"Metrics {metric_x} or {metric_y} not in leaderboard")

    fig, ax = plt.subplots(figsize=figsize)

    x = leaderboard[metric_x]
    y = leaderboard[metric_y]

    scatter = ax.scatter(x, y, c=leaderboard.index, cmap="viridis", alpha=0.6, s=50)
    ax.set_xlabel(metric_x)
    ax.set_ylabel(metric_y)
    ax.set_title(f"Efficient Frontier: {metric_x} vs {metric_y}")
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Configuration Index")

    return fig


def plot_walkforward_results(
    wf_result: WalkForwardResult,
    metric: str = "sharpe_ratio",
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot walk-forward train/test metrics across windows.

    Args:
        wf_result: Walk-forward result
        metric: Metric to plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_windows = len(wf_result.windows)
    window_ids = list(range(n_windows))

    train_values = [m.get(metric, np.nan) for m in wf_result.train_metrics]
    test_values = [m.get(metric, np.nan) for m in wf_result.test_metrics]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(window_ids, train_values, marker="o", label="Train", linewidth=2)
    ax.plot(window_ids, test_values, marker="s", label="Test (OOS)", linewidth=2)

    ax.set_xlabel("Window ID")
    ax.set_ylabel(metric)
    ax.set_title(f"Walk-Forward {metric} (Train vs Test)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_monte_carlo_distribution(
    mc_result: MonteCarloResult,
    metric: str = "sharpe_ratio",
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot Monte Carlo metric distribution.

    Args:
        mc_result: Monte Carlo result
        metric: Metric to plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if metric not in mc_result.metrics_distributions:
        raise ValueError(f"Metric {metric} not in Monte Carlo results")

    values = mc_result.metrics_distributions[metric]
    values = values[~np.isinf(values)]

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(values, bins=50, alpha=0.7, edgecolor="black")

    percentiles = mc_result.percentiles[metric]
    for p_name, p_val in percentiles.items():
        if p_name == "p50":
            ax.axvline(p_val, color="red", linestyle="--", linewidth=2, label=f"Median: {p_val:.3f}")
        elif p_name in ["p5", "p95"]:
            ax.axvline(
                p_val,
                color="orange",
                linestyle=":",
                linewidth=1.5,
                label=f"{p_name.upper()}: {p_val:.3f}",
            )

    ax.set_xlabel(metric)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Monte Carlo Distribution: {metric} ({mc_result.n_simulations} simulations)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_parameter_importance(
    leaderboard: pd.DataFrame,
    metric: str = "train_sharpe_ratio",
    top_n: int = 10,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot parameter importance based on correlation with objective metric.

    Args:
        leaderboard: Leaderboard dataframe
        metric: Target metric
        top_n: Number of top parameters to show
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if metric not in leaderboard.columns:
        raise ValueError(f"Metric {metric} not in leaderboard")

    numeric_cols = leaderboard.select_dtypes(include=[np.number]).columns
    param_cols = [c for c in numeric_cols if not c.startswith("train_") and not c.startswith("test_")]

    correlations = {}
    for col in param_cols:
        if col in leaderboard.columns:
            corr = leaderboard[col].corr(leaderboard[metric])
            if not np.isnan(corr):
                correlations[col] = abs(corr)

    sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:top_n]

    params = [p[0] for p in sorted_params]
    importances = [p[1] for p in sorted_params]

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(params, importances, color="steelblue", edgecolor="black")
    ax.set_xlabel(f"Absolute Correlation with {metric}")
    ax.set_title("Parameter Importance")
    ax.grid(True, alpha=0.3, axis="x")

    return fig


def plot_optimization_progress(
    opt_run: OptimizationRun,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot optimization progress over iterations.

    Args:
        opt_run: Optimization run
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    scores = [r.train_metrics.get(opt_run.objective_metric, np.nan) for r in opt_run.results]
    iterations = list(range(len(scores)))

    cumulative_best = []
    current_best = -np.inf if opt_run.leaderboard is not None else np.inf

    for score in scores:
        if np.isnan(score):
            cumulative_best.append(current_best)
        else:
            if len(cumulative_best) == 0:
                current_best = score
            else:
                if opt_run.leaderboard is not None:
                    current_best = max(current_best, score)
                else:
                    current_best = min(current_best, score)
            cumulative_best.append(current_best)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(iterations, scores, marker="o", alpha=0.5, label="Current Score", markersize=4)
    ax.plot(iterations, cumulative_best, linewidth=2, label="Best Score", color="red")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(opt_run.objective_metric)
    ax.set_title(f"Optimization Progress ({opt_run.method})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig
