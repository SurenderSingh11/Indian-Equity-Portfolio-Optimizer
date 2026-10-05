"""Tests for Prophet model module."""

import numpy as np
import pandas as pd

from src.model import ProphetModel, _get_indian_trading_holidays


class TestProphetModel:
    """Test Prophet model fitting, predictions, and calendar integration."""

    def test_fit(self) -> None:
        """Test fitting Prophet model on daily stock price series."""
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        price_series = pd.Series(prices, index=dates)

        model = ProphetModel()
        model.fit(price_series)

        assert model.model is not None

    def test_predict_next(self) -> None:
        """Test one-step business day forward forecast."""
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        price_series = pd.Series(prices, index=dates)

        model = ProphetModel()
        predicted_price = model.predict_next(price_series)

        assert isinstance(predicted_price, float)
        assert predicted_price > 0
        assert model.model is not None

    def test_predict_for_tickers(self) -> None:
        """Test multi-asset forecasting and return calculations."""
        dates = pd.date_range("2024-01-01", periods=100, freq="B")

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

        assert len(predictions) == 2
        assert len(predicted_returns) == 2
        assert "TICKER1" in predictions
        assert "TICKER2" in predictions

        # Check return mathematical derivation
        current_price1 = df1["Price"].iloc[-1]
        expected_return1 = (predictions["TICKER1"] - current_price1) / current_price1
        assert np.isclose(predicted_returns["TICKER1"], expected_return1, rtol=1e-5)

    def test_get_indian_trading_holidays(self) -> None:
        """Test NSE exchange calendar holiday loader."""
        holidays = _get_indian_trading_holidays(2024, 2025)

        assert isinstance(holidays, pd.DataFrame)
        assert "holiday" in holidays.columns
        assert "ds" in holidays.columns
        assert "lower_window" in holidays.columns
        assert "upper_window" in holidays.columns
        assert pd.api.types.is_datetime64_any_dtype(holidays["ds"])
        assert len(holidays) > 0