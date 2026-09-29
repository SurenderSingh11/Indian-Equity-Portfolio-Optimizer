import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.settings import MINIMUM_ALLOCATION, RISK_AVERSION


def calculate_mean_variance(
    data_dict: dict[str, pd.DataFrame],
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Calculate mean returns and covariance matrix from Returns columns.

    Args:
        data_dict: Dictionary where each key is a ticker symbol and each value
            is a DataFrame containing at least a "Returns" column representing
            periodic returns for that asset.

    Returns:
        Tuple containing:
        - mean_returns: pd.Series of mean returns for each ticker, indexed by ticker
        - cov_matrix: pd.DataFrame covariance matrix of returns across all tickers
    """
    returns_df = pd.DataFrame({ticker: df["Returns"] for ticker, df in data_dict.items()})
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    return mean_returns, cov_matrix


def optimize_portfolio_mean_variance(
    data_dict: dict[str, pd.DataFrame],
    minimum_allocation: float = MINIMUM_ALLOCATION,
    risk_aversion: float = RISK_AVERSION,
) -> pd.Series:
    """
    Optimise portfolio using mean-variance (maximise return - risk_penalty).

    Args:
        data_dict: Dictionary of DataFrames with 'Returns' column
        minimum_allocation: Minimum allocation for each asset (default: MINIMUM_ALLOCATION)
        risk_aversion: Risk-aversion coefficient (lambda) (default: RISK_AVERSION)

    Returns:
        pd.Series of optimal weights indexed by ticker, where weights sum to 1.0

    Raises:
        ValueError: If optimisation fails
    """
    mu, cov = calculate_mean_variance(data_dict)
    tickers = list(data_dict.keys())
    num_assets = len(tickers)

    # Objective: maximise return - (lambda/2) * variance
    # minimise negative of it
    def objective(weights: np.ndarray) -> float:
        port_return = float(np.dot(weights, mu))
        port_var = float(np.dot(weights.T, np.dot(cov, weights)))
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
        raise ValueError(f"Optimisation failed: {result.message}")

    return pd.Series(result.x, index=tickers)
