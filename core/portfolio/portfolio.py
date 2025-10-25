from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from core.backtest.engine import BacktestConfig
from core.portfolio.analytics import (
    compute_correlation_matrix,
    compute_portfolio_metrics,
    compute_portfolio_volatility,
    compute_risk_contributions,
)
from core.portfolio.executor import (
    SymbolBacktestResult,
    align_symbol_results,
    compute_symbol_statistics,
    run_parallel_backtests,
)
from core.portfolio.weights import (
    apply_weight_caps,
    compute_equal_weights,
    compute_risk_parity_weights,
    compute_volatility_scaled_weights,
    rebalance_weights,
)


@dataclass
class PortfolioConfig:
    """Configuration for portfolio management."""

    weighting_method: str = "equal"  # equal, volatility, risk_parity, custom
    rebalance_frequency: int = 24  # Hours between rebalances
    rebalance_threshold: float = 0.05  # Threshold for triggering rebalance
    min_weight: float = 0.0
    max_weight: float = 1.0
    target_vol: float = 0.02
    lookback_period: int = 60  # Bars for volatility/covariance calculation


@dataclass
class PortfolioResult:
    """Complete portfolio backtest results."""

    equity_curve: np.ndarray
    weights: np.ndarray
    symbols: list[str]
    symbol_results: dict[str, SymbolBacktestResult]
    returns: pd.DataFrame
    portfolio_returns: np.ndarray
    metrics: dict[str, float]
    correlation_matrix: pd.DataFrame
    rebalance_dates: list[int]


class Portfolio:
    """Portfolio manager for multi-symbol backtesting."""

    def __init__(
        self,
        symbols: list[str],
        portfolio_config: PortfolioConfig,
        backtest_config: BacktestConfig,
    ):
        """
        Initialize portfolio manager.

        Args:
            symbols: List of trading symbols
            portfolio_config: Portfolio configuration
            backtest_config: Backtest configuration
        """
        self.symbols = symbols
        self.portfolio_config = portfolio_config
        self.backtest_config = backtest_config
        self.weights = compute_equal_weights(len(symbols))

    def compute_weights(
        self,
        returns_df: pd.DataFrame,
        method: str | None = None,
    ) -> np.ndarray:
        """
        Compute portfolio weights based on specified method.

        Args:
            returns_df: DataFrame of asset returns
            method: Weighting method (overrides config if provided)

        Returns:
            Array of weights
        """
        method = method or self.portfolio_config.weighting_method
        n_assets = len(self.symbols)

        if method == "equal":
            weights = compute_equal_weights(n_assets)

        elif method == "volatility":
            weights = compute_volatility_scaled_weights(
                returns=returns_df,
                target_vol=self.portfolio_config.target_vol,
                lookback=self.portfolio_config.lookback_period,
                min_weight=self.portfolio_config.min_weight,
                max_weight=self.portfolio_config.max_weight,
            )

        elif method == "risk_parity":
            weights = compute_risk_parity_weights(
                returns=returns_df,
                lookback=self.portfolio_config.lookback_period,
                min_weight=self.portfolio_config.min_weight,
                max_weight=self.portfolio_config.max_weight,
            )

        else:
            # Default to equal weights
            weights = compute_equal_weights(n_assets)

        return weights

    def run_backtest(
        self,
        data_dict: dict[str, pd.DataFrame],
        strategy_func: Callable,
        strategy_params: dict,
    ) -> PortfolioResult:
        """
        Run portfolio backtest across all symbols.

        Args:
            data_dict: Dictionary mapping symbol to OHLCV DataFrame
            strategy_func: Function that generates signals from df and params
            strategy_params: Parameters for strategy function

        Returns:
            PortfolioResult with complete portfolio analysis
        """
        # Run individual symbol backtests
        symbol_results = run_parallel_backtests(
            symbols=self.symbols,
            data_dict=data_dict,
            strategy_func=strategy_func,
            strategy_params=strategy_params,
            config=self.backtest_config,
        )

        # Align results across symbols
        equity_df, returns_df = align_symbol_results(symbol_results)

        if equity_df.empty or returns_df.empty:
            # Return empty result
            return PortfolioResult(
                equity_curve=np.array([self.backtest_config.initial_capital]),
                weights=self.weights,
                symbols=self.symbols,
                symbol_results=symbol_results,
                returns=returns_df,
                portfolio_returns=np.array([0.0]),
                metrics={},
                correlation_matrix=pd.DataFrame(),
                rebalance_dates=[],
            )

        # Initialize portfolio tracking
        n_bars = len(equity_df)
        portfolio_equity = np.zeros(n_bars)
        portfolio_equity[0] = self.backtest_config.initial_capital

        current_weights = self.compute_weights(returns_df)
        rebalance_dates = [0]

        # Simulate portfolio evolution with rebalancing
        for i in range(1, n_bars):
            # Get recent returns for weight calculation
            if i >= self.portfolio_config.lookback_period:
                recent_returns = returns_df.iloc[max(0, i - self.portfolio_config.lookback_period):i]
            else:
                recent_returns = returns_df.iloc[:i]

            # Check if rebalancing is needed
            if i % self.portfolio_config.rebalance_frequency == 0:
                target_weights = self.compute_weights(recent_returns)
                new_weights, should_rebalance = rebalance_weights(
                    current_weights,
                    target_weights,
                    self.portfolio_config.rebalance_threshold,
                )

                if should_rebalance:
                    current_weights = new_weights
                    rebalance_dates.append(i)

            # Calculate portfolio return as weighted sum of asset returns
            bar_returns = returns_df.iloc[i].values
            portfolio_return = np.sum(current_weights * bar_returns)

            # Update portfolio equity
            portfolio_equity[i] = portfolio_equity[i - 1] * (1 + portfolio_return)

        # Calculate portfolio returns
        portfolio_returns = np.diff(portfolio_equity) / portfolio_equity[:-1]
        portfolio_returns = np.concatenate([[0.0], portfolio_returns])

        # Compute correlation matrix
        correlation_matrix = compute_correlation_matrix(returns_df)

        # Compute portfolio metrics
        metrics = compute_portfolio_metrics(
            equity_curve=portfolio_equity,
            weights=current_weights,
            returns=returns_df,
            initial_capital=self.backtest_config.initial_capital,
        )

        # Add additional portfolio-specific metrics
        cov_matrix = returns_df.cov().values
        portfolio_vol = compute_portfolio_volatility(current_weights, cov_matrix)
        risk_contributions = compute_risk_contributions(current_weights, cov_matrix)

        metrics["portfolio_volatility"] = portfolio_vol
        metrics["n_rebalances"] = len(rebalance_dates)

        return PortfolioResult(
            equity_curve=portfolio_equity,
            weights=current_weights,
            symbols=self.symbols,
            symbol_results=symbol_results,
            returns=returns_df,
            portfolio_returns=portfolio_returns,
            metrics=metrics,
            correlation_matrix=correlation_matrix,
            rebalance_dates=rebalance_dates,
        )

    def get_symbol_statistics(
        self,
        symbol_results: dict[str, SymbolBacktestResult],
    ) -> pd.DataFrame:
        """
        Get summary statistics for individual symbols.

        Args:
            symbol_results: Dictionary of symbol results

        Returns:
            DataFrame with symbol statistics
        """
        return compute_symbol_statistics(symbol_results)


def create_portfolio(
    symbols: list[str],
    weighting_method: str = "equal",
    initial_capital: float = 10000.0,
    **kwargs,
) -> Portfolio:
    """
    Create a portfolio with default configurations.

    Args:
        symbols: List of trading symbols
        weighting_method: Weighting method to use
        initial_capital: Initial capital
        **kwargs: Additional configuration parameters

    Returns:
        Configured Portfolio instance
    """
    portfolio_config = PortfolioConfig(weighting_method=weighting_method, **kwargs)
    backtest_config = BacktestConfig(initial_capital=initial_capital)

    return Portfolio(
        symbols=symbols,
        portfolio_config=portfolio_config,
        backtest_config=backtest_config,
    )
