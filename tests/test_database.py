"""Tests for database module."""

import os
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.database import get_supabase_client, save_results_to_supabase


class TestGetSupabaseClient:
    """Test Supabase client creation and environment handling."""

    @patch.dict(
        "os.environ",
        {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_KEY": "test-key"},
    )
    @patch("src.database.create_client")
    def test_get_supabase_client_with_credentials(
        self, mock_create_client: MagicMock
    ) -> None:
        """Test get_supabase_client returns client when credentials are present."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        client = get_supabase_client()

        assert client == mock_client
        mock_create_client.assert_called_once_with(
            "https://test.supabase.co", "test-key"
        )

    @patch.dict("os.environ", {}, clear=True)
    def test_get_supabase_client_raises_value_error_when_missing(self) -> None:
        """Test get_supabase_client raises ValueError when credentials are missing."""
        with pytest.raises(ValueError, match="Supabase client unavailable"):
            get_supabase_client()


class TestSaveResultsToSupabase:
    """Test saving multi-table forecasting and portfolio optimization outputs."""

    @patch("src.database.get_supabase_client")
    def test_save_results_to_supabase_full_flow(
        self, mock_get_client: MagicMock
    ) -> None:
        """Test schema-aware multi-table database ingestion."""
        mock_client = MagicMock()
        mock_schema = MagicMock()
        mock_table = MagicMock()
        mock_query = MagicMock()

        mock_get_client.return_value = mock_client
        mock_client.schema.return_value = mock_schema
        mock_schema.table.return_value = mock_table
        mock_table.upsert.return_value = mock_query
        mock_table.insert.return_value = mock_query

        test_result = {
            "schema": "prod",
            "date": date(2026, 10, 5),
            "predictions": {"RELIANCE.NS": 2500.0, "TCS.NS": 3800.0},
            "metrics": {
                "RELIANCE.NS": {"mape": 0.02, "rmse": 15.0, "mae": 10.0},
                "TCS.NS": {"mape": 0.015, "rmse": 20.0, "mae": 12.0},
            },
            "weights": {"RELIANCE.NS": 0.6, "TCS.NS": 0.4},
            "actual_prices_last_month": {
                "RELIANCE.NS": [
                    {"Date": "2026-10-01", "Close": 2480.0, "Adj Close": 2480.0}
                ]
            },
        }

        save_results_to_supabase(test_result)

        # Verify schema targeted 'prod'
        mock_client.schema.assert_called_with("prod")

        # Check forecast_outputs upsert call
        mock_schema.table.assert_any_call("forecast_outputs")
        forecast_args = mock_table.upsert.call_args_list[0][0][0]
        assert len(forecast_args) == 2
        assert forecast_args[0]["ticker"] in ("RELIANCE.NS", "TCS.NS")
        assert forecast_args[0]["execution_date"] == "2026-10-05"

        # Check model_evaluations insert call
        mock_schema.table.assert_any_call("model_evaluations")
        eval_args = mock_table.insert.call_args_list[0][0][0]
        assert len(eval_args) == 2
        assert eval_args[0]["mape"] in (0.02, 0.015)

        # Check portfolio_allocations upsert call
        mock_schema.table.assert_any_call("portfolio_allocations")
        alloc_args = mock_table.upsert.call_args_list[1][0][0]
        assert len(alloc_args) == 1
        assert alloc_args[0]["weights"] == {"RELIANCE.NS": 0.6, "TCS.NS": 0.4}

        # Check asset_prices upsert call
        mock_schema.table.assert_any_call("asset_prices")
        price_args = mock_table.upsert.call_args_list[2][0][0]
        assert len(price_args) == 1
        assert price_args[0]["ticker"] == "RELIANCE.NS"
        assert price_args[0]["close_price"] == 2480.0

    @patch("src.database.get_supabase_client")
    def test_save_results_to_supabase_empty_payload(
        self, mock_get_client: MagicMock
    ) -> None:
        """Test save_results_to_supabase handles empty input gracefully without DB calls."""
        save_results_to_supabase({})
        mock_get_client.assert_not_called()