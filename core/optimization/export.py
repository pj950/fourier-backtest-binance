"""Export optimization results to various formats."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from core.optimization.monte_carlo import MonteCarloResult
from core.optimization.runner import OptimizationRun
from core.optimization.walkforward import WalkForwardResult


def export_leaderboard_csv(
    opt_run: OptimizationRun,
    filepath: str | Path,
) -> None:
    """
    Export leaderboard to CSV.

    Args:
        opt_run: Optimization run
        filepath: Output file path
    """
    opt_run.leaderboard.to_csv(filepath, index=False)


def export_leaderboard_parquet(
    opt_run: OptimizationRun,
    filepath: str | Path,
) -> None:
    """
    Export leaderboard to Parquet.

    Args:
        opt_run: Optimization run
        filepath: Output file path
    """
    opt_run.leaderboard.to_parquet(filepath, index=False, engine="pyarrow")


def export_best_config_json(
    opt_run: OptimizationRun,
    filepath: str | Path,
) -> None:
    """
    Export best configuration to JSON.

    Args:
        opt_run: Optimization run
        filepath: Output file path
    """
    config = {
        "method": opt_run.method,
        "objective_metric": opt_run.objective_metric,
        "best_score": float(opt_run.best_score),
        "total_runtime": float(opt_run.total_runtime),
        "seed": opt_run.seed,
        "best_params": opt_run.best_params.to_dict(),
    }

    with open(filepath, "w") as f:
        json.dump(config, f, indent=2, default=_json_serializer)


def export_walkforward_results(
    wf_result: WalkForwardResult,
    output_dir: str | Path,
) -> None:
    """
    Export walk-forward results to directory.

    Args:
        wf_result: Walk-forward result
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    windows_data = []
    for window in wf_result.windows:
        windows_data.append(
            {
                "window_id": window.window_id,
                "train_start_idx": window.train_start_idx,
                "train_end_idx": window.train_end_idx,
                "test_start_idx": window.test_start_idx,
                "test_end_idx": window.test_end_idx,
            }
        )

    windows_df = pd.DataFrame(windows_data)
    windows_df.to_csv(output_path / "windows.csv", index=False)

    train_metrics_df = pd.DataFrame(wf_result.train_metrics)
    train_metrics_df["window_id"] = train_metrics_df.index
    train_metrics_df.to_csv(output_path / "train_metrics.csv", index=False)

    test_metrics_df = pd.DataFrame(wf_result.test_metrics)
    test_metrics_df["window_id"] = test_metrics_df.index
    test_metrics_df.to_csv(output_path / "test_metrics.csv", index=False)

    params_df = pd.DataFrame(wf_result.best_params_per_window)
    params_df["window_id"] = params_df.index
    params_df.to_csv(output_path / "best_params_per_window.csv", index=False)

    with open(output_path / "combined_oos_metrics.json", "w") as f:
        json.dump(wf_result.combined_oos_metrics, f, indent=2, default=_json_serializer)


def export_monte_carlo_results(
    mc_result: MonteCarloResult,
    output_dir: str | Path,
) -> None:
    """
    Export Monte Carlo results to directory.

    Args:
        mc_result: Monte Carlo result
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_simulations": mc_result.n_simulations,
        "mean_metrics": mc_result.mean_metrics,
        "std_metrics": mc_result.std_metrics,
        "percentiles": mc_result.percentiles,
    }

    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_serializer)

    for metric_name, values in mc_result.metrics_distributions.items():
        df = pd.DataFrame({"simulation_id": range(len(values)), metric_name: values})
        df.to_csv(output_path / f"distribution_{metric_name}.csv", index=False)


def export_full_optimization_results(
    opt_run: OptimizationRun,
    output_dir: str | Path,
    include_visualizations: bool = False,
) -> None:
    """
    Export complete optimization results to directory.

    Args:
        opt_run: Optimization run
        output_dir: Output directory path
        include_visualizations: If True, save visualization plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    export_leaderboard_csv(opt_run, output_path / "leaderboard.csv")
    export_leaderboard_parquet(opt_run, output_path / "leaderboard.parquet")
    export_best_config_json(opt_run, output_path / "best_config.json")

    if include_visualizations:
        from core.optimization.visualization import (
            plot_optimization_progress,
            plot_parameter_importance,
        )

        try:
            fig = plot_optimization_progress(opt_run)
            fig.savefig(output_path / "optimization_progress.png", dpi=150, bbox_inches="tight")

            fig = plot_parameter_importance(opt_run.leaderboard, f"train_{opt_run.objective_metric}")
            fig.savefig(output_path / "parameter_importance.png", dpi=150, bbox_inches="tight")
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")


def _json_serializer(obj: Any) -> Any:
    """JSON serializer for non-standard types."""
    import numpy as np

    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
