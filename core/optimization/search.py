"""Parameter search algorithms: Grid, Random, and Bayesian."""

import itertools
from typing import Any, Callable

import numpy as np

from core.optimization.params import ParamSpace, StrategyParams


class GridSearch:
    """Grid search over parameter space."""

    def __init__(
        self,
        param_spaces: dict[str, ParamSpace],
        n_points_per_param: int = 5,
        seed: int | None = None,
    ):
        """
        Initialize grid search.

        Args:
            param_spaces: Dictionary of parameter spaces
            n_points_per_param: Number of grid points per parameter
            seed: Random seed for reproducibility
        """
        self.param_spaces = param_spaces
        self.n_points_per_param = n_points_per_param
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_candidates(self) -> list[dict[str, Any]]:
        """
        Generate all grid combinations.

        Returns:
            List of parameter dictionaries
        """
        param_grids = {}

        for name, space in self.param_spaces.items():
            if space.param_type == "int":
                param_grids[name] = np.linspace(
                    space.min_val, space.max_val, self.n_points_per_param, dtype=int
                ).tolist()
            elif space.param_type == "float":
                if space.log_scale:
                    param_grids[name] = np.logspace(
                        np.log10(space.min_val),
                        np.log10(space.max_val),
                        self.n_points_per_param,
                    ).tolist()
                else:
                    param_grids[name] = np.linspace(
                        space.min_val, space.max_val, self.n_points_per_param
                    ).tolist()
            elif space.param_type == "categorical":
                param_grids[name] = space.categories

        keys = list(param_grids.keys())
        values = list(param_grids.values())

        candidates = []
        for combo in itertools.product(*values):
            candidates.append(dict(zip(keys, combo)))

        return candidates


class RandomSearch:
    """Random search over parameter space."""

    def __init__(
        self,
        param_spaces: dict[str, ParamSpace],
        n_iter: int = 100,
        seed: int | None = None,
    ):
        """
        Initialize random search.

        Args:
            param_spaces: Dictionary of parameter spaces
            n_iter: Number of random samples
            seed: Random seed for reproducibility
        """
        self.param_spaces = param_spaces
        self.n_iter = n_iter
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_candidates(self) -> list[dict[str, Any]]:
        """
        Generate random parameter combinations.

        Returns:
            List of parameter dictionaries
        """
        candidates = []

        for _ in range(self.n_iter):
            params = {}
            for name, space in self.param_spaces.items():
                if space.param_type == "int":
                    params[name] = self.rng.randint(space.min_val, space.max_val + 1)
                elif space.param_type == "float":
                    if space.log_scale:
                        log_min = np.log10(space.min_val)
                        log_max = np.log10(space.max_val)
                        params[name] = 10 ** self.rng.uniform(log_min, log_max)
                    else:
                        params[name] = self.rng.uniform(space.min_val, space.max_val)
                elif space.param_type == "categorical":
                    params[name] = self.rng.choice(space.categories)

            candidates.append(params)

        return candidates


class BayesianSearch:
    """Bayesian optimization using Gaussian Process."""

    def __init__(
        self,
        param_spaces: dict[str, ParamSpace],
        n_initial: int = 10,
        n_iter: int = 50,
        seed: int | None = None,
        acquisition: str = "ei",  # "ei", "ucb", "poi"
        kappa: float = 2.576,  # for UCB
        xi: float = 0.01,  # for EI/POI
    ):
        """
        Initialize Bayesian optimization.

        Args:
            param_spaces: Dictionary of parameter spaces
            n_initial: Number of random initial samples
            n_iter: Number of BO iterations after initial samples
            seed: Random seed for reproducibility
            acquisition: Acquisition function ("ei", "ucb", "poi")
            kappa: Kappa parameter for UCB
            xi: Xi parameter for EI/POI
        """
        self.param_spaces = param_spaces
        self.n_initial = n_initial
        self.n_iter = n_iter
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.acquisition = acquisition
        self.kappa = kappa
        self.xi = xi

        self.X_observed: list[dict[str, Any]] = []
        self.y_observed: list[float] = []
        self.param_names = list(param_spaces.keys())

    def _params_to_array(self, params: dict[str, Any]) -> np.ndarray:
        """Convert parameters to normalized array."""
        x = np.zeros(len(self.param_names))
        for i, name in enumerate(self.param_names):
            space = self.param_spaces[name]
            val = params[name]

            if space.param_type == "categorical":
                idx = space.categories.index(val)
                x[i] = idx / (len(space.categories) - 1) if len(space.categories) > 1 else 0.5
            elif space.param_type in ["int", "float"]:
                if space.log_scale:
                    log_min = np.log10(space.min_val)
                    log_max = np.log10(space.max_val)
                    x[i] = (np.log10(val) - log_min) / (log_max - log_min)
                else:
                    x[i] = (val - space.min_val) / (space.max_val - space.min_val)

        return x

    def _array_to_params(self, x: np.ndarray) -> dict[str, Any]:
        """Convert normalized array to parameters."""
        params = {}
        for i, name in enumerate(self.param_names):
            space = self.param_spaces[name]
            val = x[i]

            if space.param_type == "categorical":
                idx = int(val * (len(space.categories) - 1) + 0.5)
                idx = np.clip(idx, 0, len(space.categories) - 1)
                params[name] = space.categories[idx]
            elif space.param_type == "int":
                if space.log_scale:
                    log_min = np.log10(space.min_val)
                    log_max = np.log10(space.max_val)
                    params[name] = int(10 ** (val * (log_max - log_min) + log_min))
                else:
                    params[name] = int(val * (space.max_val - space.min_val) + space.min_val)
            elif space.param_type == "float":
                if space.log_scale:
                    log_min = np.log10(space.min_val)
                    log_max = np.log10(space.max_val)
                    params[name] = 10 ** (val * (log_max - log_min) + log_min)
                else:
                    params[name] = val * (space.max_val - space.min_val) + space.min_val

        return params

    def _gaussian_process(self, X: np.ndarray, y: np.ndarray, x_test: np.ndarray) -> tuple[float, float]:
        """Simple GP prediction using RBF kernel."""
        if len(X) == 0:
            return 0.0, 1.0

        length_scale = 0.3
        noise = 1e-5

        def rbf_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
            return np.exp(-0.5 * np.sum((x1 - x2) ** 2) / length_scale**2)

        K = np.array([[rbf_kernel(X[i], X[j]) for j in range(len(X))] for i in range(len(X))])
        K += noise * np.eye(len(X))

        k_star = np.array([rbf_kernel(x_test, X[i]) for i in range(len(X))])

        try:
            K_inv = np.linalg.inv(K)
            mu = k_star @ K_inv @ y
            sigma = 1.0 - k_star @ K_inv @ k_star
            sigma = max(sigma, 1e-10)
        except np.linalg.LinAlgError:
            mu = np.mean(y)
            sigma = 1.0

        return mu, np.sqrt(sigma)

    def _acquisition_function(self, x: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Compute acquisition function value."""
        if len(X) == 0:
            return self.rng.random()

        mu, sigma = self._gaussian_process(X, y, x)

        if self.acquisition == "ucb":
            return mu + self.kappa * sigma
        elif self.acquisition == "ei":
            from scipy.stats import norm

            best_y = np.max(y)
            z = (mu - best_y - self.xi) / (sigma + 1e-10)
            return (mu - best_y - self.xi) * norm.cdf(z) + sigma * norm.pdf(z)
        elif self.acquisition == "poi":
            from scipy.stats import norm

            best_y = np.max(y)
            z = (mu - best_y - self.xi) / (sigma + 1e-10)
            return norm.cdf(z)
        else:
            return mu

    def generate_candidates(self) -> list[dict[str, Any]]:
        """
        Generate candidates for Bayesian optimization.

        This returns initial random samples. After calling update_observations(),
        subsequent calls will suggest BO-optimized points.

        Returns:
            List of parameter dictionaries
        """
        if len(self.X_observed) < self.n_initial:
            n_samples = self.n_initial - len(self.X_observed)
            random_search = RandomSearch(self.param_spaces, n_iter=n_samples, seed=self.seed)
            return random_search.generate_candidates()

        candidates = []
        X_array = np.array([self._params_to_array(p) for p in self.X_observed])
        y_array = np.array(self.y_observed)

        for _ in range(self.n_iter):
            n_random_samples = 100
            x_samples = self.rng.random((n_random_samples, len(self.param_names)))

            acq_values = [self._acquisition_function(x, X_array, y_array) for x in x_samples]
            best_idx = np.argmax(acq_values)
            best_x = x_samples[best_idx]

            params = self._array_to_params(best_x)
            candidates.append(params)

        return candidates

    def update_observations(self, params: dict[str, Any], score: float):
        """
        Update Bayesian model with new observation.

        Args:
            params: Parameter dictionary
            score: Objective function value (higher is better)
        """
        self.X_observed.append(params)
        self.y_observed.append(score)
