"""Markowitz Mean-Variance Portfolio Optimisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.settings import MAXIMUM_ALLOCATION, MINIMUM_ALLOCATION, RISK_AVERSION


def calculate_mean_variance(
    data_dict: dict[str, pd.DataFrame],
    lookback_days: int = 252,  # ~1 trading year
) -> tuple[pd.Series, pd.DataFrame]:
    """Calculate expected mean returns and covariance matrix from historical returns."""
    filtered_data = {}
    for ticker, df in data_dict.items():
        filtered_df = df.tail(lookback_days)
        if len(filtered_df) > 0:
            filtered_data[ticker] = filtered_df

    if not filtered_data:
        filtered_data = data_dict

    returns_df = pd.DataFrame({ticker: df["Returns"] for ticker, df in filtered_data.items()})

    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()

    return mean_returns, cov_matrix


def optimize_portfolio_mean_variance(
    data_dict: dict[str, pd.DataFrame],
    minimum_allocation: float = MINIMUM_ALLOCATION,
    maximum_allocation: float = MAXIMUM_ALLOCATION,
    risk_aversion: float = RISK_AVERSION,
) -> dict[str, float]:
    """
    Optimise portfolio weights to maximize risk-adjusted return: E[R] - (lambda/2) * Var.
    """
    mu, cov = calculate_mean_variance(data_dict)
    tickers = list(data_dict.keys())
    num_assets = len(tickers)

    def objective(weights: np.ndarray) -> float:
        port_return = float(np.dot(weights, mu))
        port_var = float(np.dot(weights.T, np.dot(cov, weights)))
        # Minimise negative utility (maximise return minus variance penalty)
        return -(port_return - 0.5 * risk_aversion * port_var)

    # Weights must sum to 1.0 (100% allocation)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Min / Max allocation bounds per asset
    bounds = tuple((minimum_allocation, maximum_allocation) for _ in range(num_assets))

    initial_weights = np.array([1.0 / num_assets] * num_assets)

    result = minimize(
        objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints
    )

    if not result.success:
        raise ValueError(f"Portfolio optimization failed: {result.message}")

    weights: dict[str, float] = {
        ticker: float(weight) for ticker, weight in zip(tickers, result.x)
    }
    return weights