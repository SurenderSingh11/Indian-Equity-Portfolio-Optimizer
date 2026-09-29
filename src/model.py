"""Prophet model for one-step forward prediction."""

from __future__ import annotations

import logging

import pandas as pd
from prophet import Prophet

logger = logging.getLogger(__name__)


class ProphetModel:
    """Prophet model for forecasting stock prices."""

    def __init__(self) -> None:
        """Initialise Prophet model."""
        self.model: Prophet | None = None

    def fit(self, price_series: pd.Series) -> ProphetModel:
        """
        Fit Prophet model to price series.

        Args:
            price_series: Historical price series with datetime index

        Returns:
            Self (ProphetModel instance) for method chaining
        """
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        df = pd.DataFrame({"ds": price_series.index, "y": price_series.values})

        # Initialise and fit model
        self.model = Prophet()
        self.model.fit(df)

        return self

    def predict_next(self, price_series: pd.Series) -> float:
        """
        Fit model and predict next day's price in one step.

        Args:
            price_series: Historical price series including current day

        Returns:
            Predicted price for next day
        """

        self.fit(price_series)

        # Get the last date from the series
        last_date = price_series.index[-1]

        # Create future dataframe with next day
        future = pd.DataFrame({"ds": pd.date_range(start=last_date, periods=2, freq="D")[1:]})

        # Make prediction
        if self.model is None:
            raise RuntimeError("Model not fitted")
        forecast = self.model.predict(future)

        return float(forecast["yhat"].iloc[0])

    def predict_for_tickers(
        self,
        portfolio_data: dict[str, pd.DataFrame],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Predict prices and returns for multiple tickers.

        Args:
            portfolio_data: Dictionary mapping ticker to DataFrame with 'Price' column

        Returns:
            Tuple containing:
            - predictions: dict[str, float] mapping ticker to predicted price
            - predicted_returns: dict[str, float] mapping ticker to predicted return
        """
        predictions: dict[str, float] = {}
        predicted_returns: dict[str, float] = {}

        for ticker in portfolio_data.keys():
            # Get stock data
            df_stock = portfolio_data[ticker]

            # Get current price
            current_price = df_stock["Price"].iloc[-1]

            # Predict next day price
            predicted_price = self.predict_next(df_stock["Price"])
            predictions[ticker] = predicted_price

            # Calculate predicted return
            daily_return = (predicted_price - current_price) / current_price
            predicted_returns[ticker] = daily_return

        return predictions, predicted_returns
