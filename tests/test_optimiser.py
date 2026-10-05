"""Tests for portfolio optimisation module."""

import numpy as np
import pandas as pd
import pytest

from src.optimiser import calculate_mean_variance, optimize_portfolio_mean_variance


class TestPortfolioOptimisation:
    """Test Markowitz SLSQP optimization functions."""

    def test_calculate_mean_variance(self) -> None:
        """Test calculating mean vector and covariance matrix from historical returns."""
        dates = pd.date_range("2024-01-01", periods=100, freq="B")

        df1 = pd.DataFrame(
            {"Price": 100.0, "Returns": np.random.randn(100) * 0.01},
            index=[d.date() for d in dates],
        )
        df2 = pd.DataFrame(
            {"Price": 50.0, "Returns": np.random.randn(100) * 0.02},
            index=[d.date() for d in dates],
        )

        data_dict = {"ASSET1": df1, "ASSET2": df2}
        mean_returns, cov_matrix = calculate_mean_variance(data_dict)

        assert isinstance(mean_returns, pd.Series)
        assert isinstance(cov_matrix, pd.DataFrame)
        assert len(mean_returns) == 2
        assert cov_matrix.shape == (2, 2)

    def test_optimize_portfolio_mean_variance_bounds(self) -> None:
        """Test SLSQP optimization allocation constraint compliance."""
        dates = pd.date_range("2024-01-01", periods=100, freq="B")

        df1 = pd.DataFrame(
            {"Price": 100.0, "Returns": np.random.randn(100) * 0.01},
            index=[d.date() for d in dates],
        )
        df2 = pd.DataFrame(
            {"Price": 50.0, "Returns": np.random.randn(100) * 0.02},
            index=[d.date() for d in dates],
        )

        data_dict = {"ASSET1": df1, "ASSET2": df2}
        min_alloc, max_alloc = 0.1, 0.9

        weights = optimize_portfolio_mean_variance(
            data_dict, minimum_allocation=min_alloc, maximum_allocation=max_alloc
        )

        assert isinstance(weights, dict)
        assert np.isclose(sum(weights.values()), 1.0, rtol=1e-5)
        assert all(w >= min_alloc for w in weights.values())
        assert all(w <= max_alloc for w in weights.values())