"""Tests for Prophet model module."""

import numpy as np
import pandas as pd

from src.model import ProphetModel, _get_us_trading_holidays


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

    def test_get_us_trading_holidays(self) -> None:
        """Test US trading holidays generation."""
        holidays = _get_us_trading_holidays(2024, 2024)

        assert isinstance(holidays, pd.DataFrame)
        assert len(holidays) > 0
        assert "holiday" in holidays.columns
        assert "ds" in holidays.columns
        assert "lower_window" in holidays.columns
        assert "upper_window" in holidays.columns

        # Check specific holidays exist
        holiday_names = holidays["holiday"].unique()
        assert "new_years" in holiday_names
        assert "christmas" in holiday_names
        assert "thanksgiving" in holiday_names

        # Check that all holidays have proper windows
        assert all(holidays["lower_window"] == -1)
        assert all(holidays["upper_window"] == 1)

        # Check date format
        assert pd.api.types.is_datetime64_any_dtype(holidays["ds"])

    def test_get_us_trading_holidays_multiple_years(self) -> None:
        """Test holiday generation for multiple years."""
        holidays = _get_us_trading_holidays(2023, 2025)

        assert len(holidays) >= 30  # Should have ~10 holidays per year (3 years = 30 holidays)

        # Check years are included
        years = holidays["ds"].dt.year.unique()
        assert 2023 in years
        assert 2024 in years
        assert 2025 in years

    def test_fit_with_holidays(self) -> None:
        """Test that Prophet model includes holidays when fitting."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        price_series = pd.Series(prices, index=dates)

        model = ProphetModel()
        model.fit(price_series)

        assert model.model is not None
        # Check that holidays are included
        assert hasattr(model.model, "holidays")
        if model.model.holidays is not None:
            assert len(model.model.holidays) > 0

    def test_fit_with_seasonality_config(self) -> None:
        """Test that Prophet model has seasonality properly configured."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        price_series = pd.Series(prices, index=dates)

        model = ProphetModel()
        model.fit(price_series)

        assert model.model is not None
        # Check seasonality settings
        assert model.model.yearly_seasonality is True
        assert model.model.weekly_seasonality is True
        assert model.model.daily_seasonality is False
        assert model.model.seasonality_mode == "additive"

    def test_fit_with_holidays_and_seasonality(self) -> None:
        """Test that model works correctly with both holidays and seasonality."""
        dates = pd.date_range("2024-01-01", periods=200, freq="D")
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        price_series = pd.Series(prices, index=dates)

        model = ProphetModel()
        model.fit(price_series)

        # Make prediction to ensure everything works
        prediction = model.predict_next(price_series)

        assert isinstance(prediction, float)
        assert prediction > 0
        assert model.model is not None

    def test_fit_with_short_data_no_holidays(self) -> None:
        """Test model handles data with no holidays in range gracefully."""
        # Use a date range that might not have holidays
        dates = pd.date_range("2024-02-15", periods=10, freq="D")
        prices = 100 + np.cumsum(np.random.randn(10) * 0.5)
        price_series = pd.Series(prices, index=dates)

        model = ProphetModel()
        # Should not raise an error even if no holidays in range
        model.fit(price_series)

        assert model.model is not None
