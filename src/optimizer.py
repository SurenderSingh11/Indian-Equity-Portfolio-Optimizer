from typing import Dict
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.settings import MINIMUM_ALLOCATION, RISK_AVERSION


def calculate_mean_variance(data_dict: Dict[str, pd.DataFrame]):
    """Calculate mean returns and covariance matrix from Returns columns."""
    returns_df = pd.DataFrame({ticker: df["Returns"] for ticker, df in data_dict.items()})
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    return mean_returns, cov_matrix


def optimize_portfolio_mean_variance(
    data_dict: Dict[str, pd.DataFrame],
    minimum_allocation: float = MINIMUM_ALLOCATION,
    risk_aversion: float = RISK_AVERSION,
) -> pd.Series:
    """
    Optimize portfolio using mean-variance (maximize return - risk_penalty).

    Args:
        data_dict: Dictionary of DataFrames with 'Returns' column
        minimum_allocation: Minimum allocation for each asset
        risk_aversion: Risk-aversion coefficient (lambda)

    Returns:
        pd.Series of optimal weights indexed by ticker
    """
    mu, cov = calculate_mean_variance(data_dict)
    tickers = list(data_dict.keys())
    num_assets = len(tickers)

    # Objective: maximize return - (lambda/2) * variance
    # minimize negative of it
    def objective(weights: np.ndarray) -> float:
        port_return = np.dot(weights, mu)
        port_var = np.dot(weights.T, np.dot(cov, weights))
        return -(port_return - 0.5 * risk_aversion * port_var)

    # Constraint: sum(weights) == 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    # Bounds: enforce minimum allocation per asset
    bounds = tuple((minimum_allocation, 1.0) for _ in range(num_assets))

    # Initial guess: equal weights
    initial_weights = np.array([1 / num_assets] * num_assets)

    # Run optimizer
    result = minimize(
        objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints
    )

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    return pd.Series(result.x, index=tickers)
