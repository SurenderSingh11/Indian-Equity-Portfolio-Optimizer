"""Tests for data module."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from src.data import extract_data, preprocess_data, append_predictions


class TestData:
    """Test data extraction and preprocessing."""

    def test_extract_data(self) -> None:
        """Test extracting historical data."""
        tickers = ["KO", "BBVA"]
        data = extract_data(tickers, start_date="2024-01-01")

        assert isinstance(data, dict)
        assert len(data) > 0
        for ticker in tickers:
            if ticker in data:
                assert isinstance(data[ticker], pd.DataFrame)
                assert "Price" in data[ticker].columns
                assert "Returns" in data[ticker].columns
                assert data[ticker].index.name == "Date"
                # Check that index is date type
                assert all(isinstance(d, date) for d in data[ticker].index)

    def test_preprocess_data(self) -> None:
        """Test preprocessing data - aligns dates across tickers."""
        # Create sample data with overlapping dates
        dates1 = pd.date_range("2024-01-01", periods=10, freq="D")
        dates2 = pd.date_range("2024-01-03", periods=8, freq="D")  # Partial overlap

        data1 = pd.DataFrame(
            {
                "Price": np.random.randn(10) * 10 + 100,
                "Returns": np.random.randn(10) * 0.01,
            },
            index=dates1,
        )
        data2 = pd.DataFrame(
            {
                "Price": np.random.randn(8) * 10 + 50,
                "Returns": np.random.randn(8) * 0.02,
            },
            index=dates2,
        )

        data_dict = {"TICKER1": data1, "TICKER2": data2}
        aligned = preprocess_data(data_dict)

        assert isinstance(aligned, dict)
        assert len(aligned) == 2
        assert "TICKER1" in aligned
        assert "TICKER2" in aligned

        # Check that both DataFrames have the same index (common dates)
        assert list(aligned["TICKER1"].index) == list(aligned["TICKER2"].index)

        # Check that indexes are date type
        assert all(isinstance(d, date) for d in aligned["TICKER1"].index)
        assert all(isinstance(d, date) for d in aligned["TICKER2"].index)

        # Check columns are preserved
        assert "Price" in aligned["TICKER1"].columns
        assert "Returns" in aligned["TICKER1"].columns
        assert "Price" in aligned["TICKER2"].columns
        assert "Returns" in aligned["TICKER2"].columns

        # Common dates should be intersection of both ranges
        dates1_set = {d.date() for d in dates1}
        dates2_set = {d.date() for d in dates2}
        expected_common_dates = sorted(dates1_set & dates2_set)
        assert len(aligned["TICKER1"]) == len(expected_common_dates)

    def test_extract_data_with_end_date(self) -> None:
        """Test extracting data with end_date filter."""
        tickers = ["KO"]
        end_date = "2024-06-01"
        data = extract_data(tickers, start_date="2024-01-01", end_date=end_date)

        assert isinstance(data, dict)
        if tickers[0] in data:
            assert isinstance(data[tickers[0]], pd.DataFrame)
            # Check that all dates are <= end_date
            if len(data[tickers[0]]) > 0:
                assert all(
                    pd.Timestamp(d) <= pd.Timestamp(end_date) for d in data[tickers[0]].index
                )

    def test_preprocess_data_empty(self) -> None:
        """Test preprocessing with empty data."""
        aligned = preprocess_data({})
        assert isinstance(aligned, dict)
        assert len(aligned) == 0

    def test_append_predictions(self) -> None:
        """Test appending predictions to portfolio data."""
        # Create sample data
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df1 = pd.DataFrame(
            {
                "Price": [100.0, 101.0, 102.0, 103.0, 104.0],
                "Returns": [0.01, 0.01, 0.01, 0.01, 0.01],
            },
            index=[d.date() for d in dates],
        )
        df2 = pd.DataFrame(
            {
                "Price": [50.0, 51.0, 52.0, 53.0, 54.0],
                "Returns": [0.02, 0.02, 0.02, 0.02, 0.02],
            },
            index=[d.date() for d in dates],
        )

        portfolio_data = {"TICKER1": df1, "TICKER2": df2}
        predictions = {"TICKER1": 105.0, "TICKER2": 55.0}
        predicted_returns = {"TICKER1": 0.0096, "TICKER2": 0.0185}

        updated = append_predictions(portfolio_data, predictions, predicted_returns)

        assert isinstance(updated, dict)
        assert len(updated) == 2

        # Check TICKER1
        assert len(updated["TICKER1"]) == 6  # Original 5 + 1 prediction
        assert updated["TICKER1"].iloc[-1]["Price"] == 105.0
        assert updated["TICKER1"].iloc[-1]["Returns"] == 0.0096
        # Check prediction date is next day
        last_date = dates[-1].date()
        prediction_date = updated["TICKER1"].index[-1]
        assert prediction_date == last_date + timedelta(days=1)

        # Check TICKER2
        assert len(updated["TICKER2"]) == 6
        assert updated["TICKER2"].iloc[-1]["Price"] == 55.0
        assert updated["TICKER2"].iloc[-1]["Returns"] == 0.0185

    def test_append_predictions_preserves_original_data(self) -> None:
        """Test that appending predictions doesn't modify original DataFrames."""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "Price": [100.0, 101.0, 102.0],
                "Returns": [0.01, 0.01, 0.01],
            },
            index=[d.date() for d in dates],
        )

        portfolio_data = {"TICKER1": df.copy()}
        original_length = len(df)

        updated = append_predictions(
            portfolio_data,
            {"TICKER1": 103.0},
            {"TICKER1": 0.01},
        )

        # Original DataFrame should be unchanged
        assert len(df) == original_length
        # Updated DataFrame should have one more row
        assert len(updated["TICKER1"]) == original_length + 1
