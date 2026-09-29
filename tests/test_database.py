"""Tests for database module."""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.database import get_supabase_client, save_results_to_supabase
from src.settings import SUPABASE_TABLE_NAME


class TestGetSupabaseClient:
    """Test Supabase client creation."""

    @patch.dict(
        "os.environ", {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_KEY": "test-key"}
    )
    @patch("src.database.create_client")
    def test_get_supabase_client_with_credentials(self, mock_create_client: MagicMock) -> None:
        """Test get_supabase_client returns client when credentials are available."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        result = get_supabase_client()

        assert result is not None
        assert result == mock_client
        mock_create_client.assert_called_once_with("https://test.supabase.co", "test-key")

    @patch.dict("os.environ", {}, clear=True)
    def test_get_supabase_client_without_url(self) -> None:
        """Test get_supabase_client returns None when URL is missing."""
        result = get_supabase_client()
        assert result is None

    @patch.dict("os.environ", {"SUPABASE_URL": "https://test.supabase.co"}, clear=True)
    def test_get_supabase_client_without_key(self) -> None:
        """Test get_supabase_client returns None when key is missing."""
        result = get_supabase_client()
        assert result is None

    @patch.dict("os.environ", {}, clear=True)
    def test_get_supabase_client_without_credentials(self) -> None:
        """Test get_supabase_client returns None when both credentials are missing."""
        result = get_supabase_client()
        assert result is None


class TestSaveResultsToSupabase:
    """Test saving results to Supabase."""

    @patch("src.database.get_supabase_client")
    def test_save_results_to_supabase_success(self, mock_get_client: MagicMock) -> None:
        """Test successfully saving results to Supabase."""
        # Setup mocks
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_insert = MagicMock()
        mock_execute = MagicMock()

        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_insert.execute.return_value = mock_execute
        mock_get_client.return_value = mock_client

        # Test data
        result = {
            "predictions": {"AAPL": 150.25, "MSFT": 380.50},
            "predicted_returns": {"AAPL": 0.02, "MSFT": 0.015},
            "weights": {"AAPL": 0.4, "MSFT": 0.6},
        }

        # Call function
        save_results_to_supabase(result)

        # Verify calls
        mock_get_client.assert_called_once()
        mock_client.table.assert_called_once_with(SUPABASE_TABLE_NAME)
        mock_table.insert.assert_called_once()

        # Check that insert was called with correct structure
        insert_call_args = mock_table.insert.call_args[0][0]
        assert len(insert_call_args) == 2  # Two stocks

        # Check first row structure
        first_row = insert_call_args[0]
        assert "id" in first_row
        assert "created_at" in first_row
        assert "stock" in first_row
        assert "price" in first_row
        assert "daily_return" in first_row
        assert "portfolio_weight" in first_row

        # Check ID is a valid UUID string
        uuid.UUID(first_row["id"])  # Will raise if invalid UUID

        # Check values
        assert first_row["stock"] in ("AAPL", "MSFT")
        if first_row["stock"] == "AAPL":
            assert first_row["price"] == 150.25
            assert first_row["daily_return"] == 0.02
            assert first_row["portfolio_weight"] == 0.4
        else:
            assert first_row["price"] == 380.50
            assert first_row["daily_return"] == 0.015
            assert first_row["portfolio_weight"] == 0.6

        mock_insert.execute.assert_called_once()

    @patch("src.database.get_supabase_client")
    def test_save_results_to_supabase_no_client(self, mock_get_client: MagicMock) -> None:
        """Test save_results_to_supabase raises error when client is None."""
        mock_get_client.return_value = None

        result = {
            "predictions": {"AAPL": 150.25},
            "predicted_returns": {"AAPL": 0.02},
            "weights": {"AAPL": 1.0},
        }

        with pytest.raises(ValueError, match="Supabase client not available"):
            save_results_to_supabase(result)

    @patch("src.database.get_supabase_client")
    def test_save_results_to_supabase_no_predictions(self, mock_get_client: MagicMock) -> None:
        """Test save_results_to_supabase returns early when no predictions."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        result = {
            "predictions": {},
            "predicted_returns": {},
            "weights": {},
        }

        save_results_to_supabase(result)

        # Should not attempt to insert
        mock_client.table.assert_not_called()

    @patch("src.database.get_supabase_client")
    def test_save_results_to_supabase_missing_keys(self, mock_get_client: MagicMock) -> None:
        """Test save_results_to_supabase handles missing keys gracefully."""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_insert = MagicMock()
        mock_execute = MagicMock()

        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_insert.execute.return_value = mock_execute
        mock_get_client.return_value = mock_client

        # Missing predicted_returns and weights
        result = {
            "predictions": {"AAPL": 150.25},
        }

        save_results_to_supabase(result)

        # Should still work, using defaults
        insert_call_args = mock_table.insert.call_args[0][0]
        assert len(insert_call_args) == 1
        assert insert_call_args[0]["stock"] == "AAPL"
        assert insert_call_args[0]["price"] == 150.25
        assert insert_call_args[0]["daily_return"] == 0.0  # Default
        assert insert_call_args[0]["portfolio_weight"] == 0.0  # Default

    @patch("src.database.get_supabase_client")
    def test_save_results_to_supabase_insert_failure(self, mock_get_client: MagicMock) -> None:
        """Test save_results_to_supabase handles insertion failure."""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_insert = MagicMock()

        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_insert.execute.side_effect = Exception("Database connection error")
        mock_get_client.return_value = mock_client

        result = {
            "predictions": {"AAPL": 150.25},
            "predicted_returns": {"AAPL": 0.02},
            "weights": {"AAPL": 1.0},
        }

        # Should propagate the exception
        with pytest.raises(Exception, match="Database connection error"):
            save_results_to_supabase(result)

    @patch("src.database.get_supabase_client")
    def test_save_results_to_supabase_row_structure(self, mock_get_client: MagicMock) -> None:
        """Test that rows have correct structure and data types."""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_insert = MagicMock()
        mock_execute = MagicMock()

        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_insert.execute.return_value = mock_execute
        mock_get_client.return_value = mock_client

        result = {
            "predictions": {"TSLA": 250.75},
            "predicted_returns": {"TSLA": -0.01},
            "weights": {"TSLA": 0.25},
        }

        save_results_to_supabase(result)

        insert_call_args = mock_table.insert.call_args[0][0]
        row = insert_call_args[0]

        # Check all required fields are present
        assert "id" in row
        assert "created_at" in row
        assert "stock" in row
        assert "price" in row
        assert "daily_return" in row
        assert "portfolio_weight" in row

        # Check data types
        assert isinstance(row["id"], str)
        assert isinstance(row["created_at"], str)
        assert isinstance(row["stock"], str)
        assert isinstance(row["price"], float)
        assert isinstance(row["daily_return"], float)
        assert isinstance(row["portfolio_weight"], float)

        # Check ID is a valid UUID
        uuid.UUID(row["id"])  # Will raise if invalid UUID

        # Check values
        assert row["stock"] == "TSLA"
        assert row["price"] == 250.75
        assert row["daily_return"] == -0.01
        assert row["portfolio_weight"] == 0.25
