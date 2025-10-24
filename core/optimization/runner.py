"""Optimization runner for batch parameter search and evaluation."""

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.optimization.params import StrategyParams
from core.optimization.search import BayesianSearch, GridSearch, RandomSearch
from core.optimization.walkforward import (
    WalkForwardResult,
    WalkForwardWindow,
    compute_combined_oos_metrics,
    create_walkforward_windows,
    split_data_by_window,
)


@dataclass
class OptimizationResult:
    """Results from parameter optimization run."""

    params: StrategyParams
    train_metrics: dict[str, float]
    test_metrics: dict[str, float] | None
    runtime_seconds: float
    param_id: int
    window_id: int | None = None


@dataclass
class OptimizationRun:
    """Complete optimization run with all results."""

    method: str
    results: list[OptimizationResult]
    best_params: StrategyParams
    best_score: float
    leaderboard: pd.DataFrame
    total_runtime: float
    objective_metric: str
    seed: int | None


class OptimizationRunner:
    """Runner for batch optimization experiments."""

    def __init__(
        self,
        objective_function: Callable[[StrategyParams, pd.DataFrame], dict[str, float]],
        objective_metric: str = "sharpe_ratio",
        maximize: bool = True,
        seed: int | None = None,
    ):
        """
        Initialize optimization runner.

        Args:
            objective_function: Function that takes params and data, returns metrics dict
            objective_metric: Metric to optimize
            maximize: If True, maximize metric; if False, minimize
            seed: Random seed for reproducibility
        """
        self.objective_function = objective_function
        self.objective_metric = objective_metric
        self.maximize = maximize
        self.seed = seed

    def run_grid_search(
        self,
        param_spaces: dict,
        data: pd.DataFrame,
        n_points_per_param: int = 5,
        verbose: bool = True,
    ) -> OptimizationRun:
        """
        Run grid search optimization.

        Args:
            param_spaces: Dictionary of parameter spaces
            data: Input data
            n_points_per_param: Grid points per parameter
            verbose: Print progress

        Returns:
            OptimizationRun with results
        """
        start_time = time.time()

        grid_search = GridSearch(param_spaces, n_points_per_param, seed=self.seed)
        candidates = grid_search.generate_candidates()

        results = []
        for i, param_dict in enumerate(candidates):
            if verbose and (i + 1) % 10 == 0:
                print(f"Evaluating candidate {i + 1}/{len(candidates)}")

            result = self._evaluate_params(param_dict, data, i, window_id=None)
            results.append(result)

        total_runtime = time.time() - start_time

        best_result = self._find_best_result(results)
        leaderboard = self._create_leaderboard(results)

        return OptimizationRun(
            method="grid_search",
            results=results,
            best_params=best_result.params,
            best_score=best_result.train_metrics.get(self.objective_metric, 0.0),
            leaderboard=leaderboard,
            total_runtime=total_runtime,
            objective_metric=self.objective_metric,
            seed=self.seed,
        )

    def run_random_search(
        self,
        param_spaces: dict,
        data: pd.DataFrame,
        n_iter: int = 100,
        verbose: bool = True,
    ) -> OptimizationRun:
        """
        Run random search optimization.

        Args:
            param_spaces: Dictionary of parameter spaces
            data: Input data
            n_iter: Number of random samples
            verbose: Print progress

        Returns:
            OptimizationRun with results
        """
        start_time = time.time()

        random_search = RandomSearch(param_spaces, n_iter, seed=self.seed)
        candidates = random_search.generate_candidates()

        results = []
        for i, param_dict in enumerate(candidates):
            if verbose and (i + 1) % 10 == 0:
                print(f"Evaluating candidate {i + 1}/{len(candidates)}")

            result = self._evaluate_params(param_dict, data, i, window_id=None)
            results.append(result)

        total_runtime = time.time() - start_time

        best_result = self._find_best_result(results)
        leaderboard = self._create_leaderboard(results)

        return OptimizationRun(
            method="random_search",
            results=results,
            best_params=best_result.params,
            best_score=best_result.train_metrics.get(self.objective_metric, 0.0),
            leaderboard=leaderboard,
            total_runtime=total_runtime,
            objective_metric=self.objective_metric,
            seed=self.seed,
        )

    def run_bayesian_search(
        self,
        param_spaces: dict,
        data: pd.DataFrame,
        n_initial: int = 10,
        n_iter: int = 50,
        verbose: bool = True,
    ) -> OptimizationRun:
        """
        Run Bayesian optimization.

        Args:
            param_spaces: Dictionary of parameter spaces
            data: Input data
            n_initial: Number of initial random samples
            n_iter: Number of BO iterations
            verbose: Print progress

        Returns:
            OptimizationRun with results
        """
        start_time = time.time()

        bayesian_search = BayesianSearch(
            param_spaces, n_initial=n_initial, n_iter=n_iter, seed=self.seed
        )

        results = []
        param_id = 0

        initial_candidates = bayesian_search.generate_candidates()
        for i, param_dict in enumerate(initial_candidates):
            if verbose:
                print(f"Initial sample {i + 1}/{len(initial_candidates)}")

            result = self._evaluate_params(param_dict, data, param_id, window_id=None)
            results.append(result)

            score = result.train_metrics.get(self.objective_metric, 0.0)
            if not self.maximize:
                score = -score
            bayesian_search.update_observations(param_dict, score)

            param_id += 1

        bo_candidates = bayesian_search.generate_candidates()
        for i, param_dict in enumerate(bo_candidates):
            if verbose:
                print(f"BO iteration {i + 1}/{len(bo_candidates)}")

            result = self._evaluate_params(param_dict, data, param_id, window_id=None)
            results.append(result)

            score = result.train_metrics.get(self.objective_metric, 0.0)
            if not self.maximize:
                score = -score
            bayesian_search.update_observations(param_dict, score)

            param_id += 1

        total_runtime = time.time() - start_time

        best_result = self._find_best_result(results)
        leaderboard = self._create_leaderboard(results)

        return OptimizationRun(
            method="bayesian_search",
            results=results,
            best_params=best_result.params,
            best_score=best_result.train_metrics.get(self.objective_metric, 0.0),
            leaderboard=leaderboard,
            total_runtime=total_runtime,
            objective_metric=self.objective_metric,
            seed=self.seed,
        )

    def run_walkforward(
        self,
        param_spaces: dict,
        data: pd.DataFrame,
        train_size: int,
        test_size: int,
        step_size: int | None = None,
        anchored: bool = False,
        search_method: str = "random",
        n_candidates: int = 50,
        verbose: bool = True,
    ) -> tuple[OptimizationRun, WalkForwardResult]:
        """
        Run walk-forward analysis with parameter optimization.

        Args:
            param_spaces: Dictionary of parameter spaces
            data: Input data
            train_size: Training window size
            test_size: Test window size
            step_size: Step size for rolling window
            anchored: Use anchored walk-forward
            search_method: "grid", "random", or "bayesian"
            n_candidates: Number of candidates per window
            verbose: Print progress

        Returns:
            Tuple of (OptimizationRun, WalkForwardResult)
        """
        start_time = time.time()

        n_bars = len(data)
        windows = create_walkforward_windows(n_bars, train_size, test_size, step_size, anchored)

        if verbose:
            print(f"Created {len(windows)} walk-forward windows")

        all_results = []
        best_params_per_window = []
        train_metrics_list = []
        test_metrics_list = []
        oos_equity_curves = []

        for window in windows:
            if verbose:
                print(f"Window {window.window_id + 1}/{len(windows)}")

            train_df, test_df = split_data_by_window(data, window)

            if search_method == "grid":
                n_points = int(n_candidates ** (1 / len(param_spaces)))
                window_run = self.run_grid_search(param_spaces, train_df, n_points, verbose=False)
            elif search_method == "bayesian":
                n_initial = min(10, n_candidates // 2)
                n_iter = n_candidates - n_initial
                window_run = self.run_bayesian_search(
                    param_spaces, train_df, n_initial, n_iter, verbose=False
                )
            else:
                window_run = self.run_random_search(param_spaces, train_df, n_candidates, verbose=False)

            best_params = window_run.best_params
            best_params_per_window.append(best_params.to_dict())

            train_metrics = self.objective_function(best_params, train_df)
            train_metrics_list.append(train_metrics)

            test_metrics = self.objective_function(best_params, test_df)
            test_metrics_list.append(test_metrics)

            for result in window_run.results:
                result.window_id = window.window_id
                all_results.append(result)

            if verbose:
                print(
                    f"  Train {self.objective_metric}: {train_metrics.get(self.objective_metric, 0.0):.4f}"
                )
                print(
                    f"  Test {self.objective_metric}: {test_metrics.get(self.objective_metric, 0.0):.4f}"
                )

        combined_oos_metrics = {}
        if test_metrics_list:
            avg_metrics = {}
            for key in test_metrics_list[0].keys():
                values = [m[key] for m in test_metrics_list if key in m]
                if values:
                    avg_metrics[key] = float(np.mean(values))
            combined_oos_metrics = avg_metrics

        total_runtime = time.time() - start_time

        best_result = self._find_best_result(all_results)
        leaderboard = self._create_leaderboard(all_results)

        opt_run = OptimizationRun(
            method=f"walkforward_{search_method}",
            results=all_results,
            best_params=best_result.params,
            best_score=best_result.train_metrics.get(self.objective_metric, 0.0),
            leaderboard=leaderboard,
            total_runtime=total_runtime,
            objective_metric=self.objective_metric,
            seed=self.seed,
        )

        wf_result = WalkForwardResult(
            windows=windows,
            train_metrics=train_metrics_list,
            test_metrics=test_metrics_list,
            best_params_per_window=best_params_per_window,
            oos_equity_curves=oos_equity_curves,
            combined_oos_metrics=combined_oos_metrics,
        )

        return opt_run, wf_result

    def _evaluate_params(
        self,
        param_dict: dict[str, Any],
        data: pd.DataFrame,
        param_id: int,
        window_id: int | None,
    ) -> OptimizationResult:
        """Evaluate a single parameter configuration."""
        start_time = time.time()

        params = StrategyParams.from_dict(param_dict)

        try:
            metrics = self.objective_function(params, data)
        except Exception as e:
            print(f"Error evaluating params: {e}")
            metrics = {self.objective_metric: -np.inf if self.maximize else np.inf}

        runtime = time.time() - start_time

        return OptimizationResult(
            params=params,
            train_metrics=metrics,
            test_metrics=None,
            runtime_seconds=runtime,
            param_id=param_id,
            window_id=window_id,
        )

    def _find_best_result(self, results: list[OptimizationResult]) -> OptimizationResult:
        """Find best result based on objective metric."""
        if not results:
            raise ValueError("No results to evaluate")

        if self.maximize:
            return max(results, key=lambda r: r.train_metrics.get(self.objective_metric, -np.inf))
        else:
            return min(results, key=lambda r: r.train_metrics.get(self.objective_metric, np.inf))

    def _create_leaderboard(self, results: list[OptimizationResult]) -> pd.DataFrame:
        """Create leaderboard dataframe from results."""
        rows = []
        for result in results:
            row = {
                "param_id": result.param_id,
                "window_id": result.window_id,
                "runtime_seconds": result.runtime_seconds,
            }
            row.update(result.params.to_dict())
            row.update({f"train_{k}": v for k, v in result.train_metrics.items()})
            if result.test_metrics:
                row.update({f"test_{k}": v for k, v in result.test_metrics.items()})
            rows.append(row)

        df = pd.DataFrame(rows)

        sort_col = f"train_{self.objective_metric}"
        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=not self.maximize)

        return df
