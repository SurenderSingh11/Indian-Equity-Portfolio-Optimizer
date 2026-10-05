"""Tests for extractor module."""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.extractor import extract_data


class TestExtractor:
    """Test market data extraction via direct v8 chart API."""

    @patch("requests.Session.get")
    def test_extract_data_success(self, mock_get: MagicMock) -> None:
        """Test successfully parsing raw Yahoo Finance v8 JSON payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "chart": {
                "result": [
                    {
                        "timestamps": [1700000000, 1700086400],
                        "indicators": {
                            "quote": [{"close": [100.0, 105.0]}],
                            "adjclose": [{"adjclose": [100.0, 105.0]}],
                        },
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        tickers = ["RELIANCE.NS"]
        data = extract_data(tickers, start_date="2024-01-01", end_date="2024-01-05")

        assert isinstance(data, dict)
        assert "RELIANCE.NS" in data
        df = data["RELIANCE.NS"]
        assert isinstance(df, pd.DataFrame)
        assert "Price" in df.columns
        assert "Returns" in df.columns
        assert len(df) > 0

    @patch("requests.Session.get")
    def test_extract_data_rate_limit_and_failure(
        self, mock_get: MagicMock
    ) -> None:
        """Test resilience to 429 rate limit responses returning empty result set."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        data = extract_data(["INVALID.NS"], start_date="2024-01-01", end_date="2024-01-05")

        assert isinstance(data, dict)
        assert len(data) == 0