"""Tests for Prophet model module."""

import pytest
import pandas as pd
import numpy as np

from src.model import ProphetModel


class TestProphetModel:
    """Test Prophet model."""

    def test_prophet_model_init(self) -> None:
        """Test Prophet model initialization."""
        model = ProphetModel()
        assert model.model is None

    def test_fit(self) -> None:
        """Test fitting Prophet model."""
        # Create sample time series
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)  # Random walk
        price_series = pd.Series(prices, index=dates)

        model = ProphetModel()
        model.fit(price_series)

        assert model.model is not None

    def test_predict_next(self) -> None:
        """Test predict_next method."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        price_series = pd.Series(prices, index=dates)

        model = ProphetModel()
        predicted_price = model.predict_next(price_series)

        assert isinstance(predicted_price, float)
        assert predicted_price > 0
        assert model.model is not None  # Model should be fitted

    def test_predict_next_multiple_calls(self) -> None:
        """Test predict_next can be called multiple times."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        price_series = pd.Series(prices, index=dates)

        model = ProphetModel()

        # First call
        predicted_price1 = model.predict_next(price_series)
        assert isinstance(predicted_price1, float)

        # Second call with same data
        predicted_price2 = model.predict_next(price_series)
        assert isinstance(predicted_price2, float)

    def test_predict_for_tickers(self) -> None:
        """Test predict_for_tickers method with multiple tickers."""
        from datetime import date

        dates = pd.date_range("2024-01-01", periods=100, freq="D")

        # Create DataFrames with Price column (as expected by predict_for_tickers)
        df1 = pd.DataFrame(
            {
                "Price": 100 + np.cumsum(np.random.randn(100) * 0.5),
                "Returns": np.random.randn(100) * 0.01,
            },
            index=[d.date() for d in dates],
        )
        df2 = pd.DataFrame(
            {
                "Price": 50 + np.cumsum(np.random.randn(100) * 0.3),
                "Returns": np.random.randn(100) * 0.02,
            },
            index=[d.date() for d in dates],
        )

        portfolio_data = {"TICKER1": df1, "TICKER2": df2}

        model = ProphetModel()
        predictions, predicted_returns = model.predict_for_tickers(portfolio_data)

        assert isinstance(predictions, dict)
        assert isinstance(predicted_returns, dict)
        assert len(predictions) == 2
        assert len(predicted_returns) == 2
        assert "TICKER1" in predictions
        assert "TICKER2" in predictions
        assert "TICKER1" in predicted_returns
        assert "TICKER2" in predicted_returns

        # Check predictions are floats and positive
        assert isinstance(predictions["TICKER1"], float)
        assert isinstance(predictions["TICKER2"], float)
        assert predictions["TICKER1"] > 0
        assert predictions["TICKER2"] > 0

        # Check predicted returns are floats
        assert isinstance(predicted_returns["TICKER1"], float)
        assert isinstance(predicted_returns["TICKER2"], float)

        # Check that predicted return is calculated correctly
        current_price1 = df1["Price"].iloc[-1]
        expected_return1 = (predictions["TICKER1"] - current_price1) / current_price1
        assert np.isclose(predicted_returns["TICKER1"], expected_return1, rtol=1e-5)
