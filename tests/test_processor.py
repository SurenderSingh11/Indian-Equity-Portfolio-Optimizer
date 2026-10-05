"""Tests for data processing and pipeline assembly utilities."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.processor import append_predictions, collect_recent_prices, preprocess_data


class TestProcessor:
    """Test data alignment, prediction appending, and price window collection."""

    def test_preprocess_data(self) -> None:
        """Test aligning market data across multiple assets onto common trading days."""
        dates1 = pd.date_range("2024-01-01", periods=10, freq="B")
        dates2 = pd.date_range("2024-01-03", periods=8, freq="B")

        df1 = pd.DataFrame(
            {"Price": 100.0, "Returns": 0.01}, index=[d.date() for d in dates1]
        )
        df2 = pd.DataFrame(
            {"Price": 50.0, "Returns": 0.02}, index=[d.date() for d in dates2]
        )

        aligned = preprocess_data({"TICKER1": df1, "TICKER2": df2})

        assert "TICKER1" in aligned
        assert "TICKER2" in aligned
        assert list(aligned["TICKER1"].index) == list(aligned["TICKER2"].index)
        assert all(isinstance(d, date) for d in aligned["TICKER1"].index)

    def test_append_predictions(self) -> None:
        """Test appending one-step forward forecast row onto asset DataFrames."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df1 = pd.DataFrame(
            {"Price": [100, 101, 102, 103, 104], "Returns": [0.01] * 5},
            index=[d.date() for d in dates],
        )

        portfolio_data = {"TICKER1": df1}
        predictions = {"TICKER1": 105.0}
        predicted_returns = {"TICKER1": 0.0096}

        updated = append_predictions(portfolio_data, predictions, predicted_returns)

        assert len(updated["TICKER1"]) == 6
        assert updated["TICKER1"].iloc[-1]["Price"] == 105.0
        assert updated["TICKER1"].iloc[-1]["Returns"] == 0.0096

    def test_collect_recent_prices(self) -> None:
        """Test collecting trailing historical price series for display."""
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        df = pd.DataFrame(
            {"Price": np.linspace(100, 140, num=40), "Returns": 0.01},
            index=[d.date() for d in dates],
        )

        recent_prices = collect_recent_prices({"TICKER1": df}, days=30)

        assert "TICKER1" in recent_prices
        assert isinstance(recent_prices["TICKER1"], list)
        assert len(recent_prices["TICKER1"]) > 0
        assert recent_prices["TICKER1"][-1] == float(df.iloc[-1]["Price"])