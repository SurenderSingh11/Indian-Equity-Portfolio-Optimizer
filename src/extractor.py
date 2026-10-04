"""Data extraction module for fetching stock data from Yahoo Finance API directly."""

import logging
import time
import pandas as pd
import requests

from .settings import END_DATE, START_DATE

logger = logging.getLogger(__name__)


def _fetch_ticker_direct_api(
    ticker: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch historical daily data for a ticker directly from Yahoo Finance v8 Chart API."""
    # Convert string dates to epoch timestamps
    period1 = int(pd.Timestamp(start_date).timestamp())
    period2 = int(pd.Timestamp(end_date).timestamp())

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }

    session = requests.Session()

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                result = data.get("chart", {}).get("result")
                if not result:
                    logger.warning(f"No chart result returned for {ticker}.")
                    return pd.DataFrame()

                timestamps = result[0].get("timestamps") or result[0].get("timestamp", [])
                indicators = result[0].get("indicators", {})
                quote = indicators.get("quote", [{}])[0]
                adjclose = indicators.get("adjclose", [{}])[0].get("adjclose")

                # Prefer Adj Close, fallback to Close
                close_prices = adjclose if adjclose else quote.get("close", [])

                if not timestamps or not close_prices:
                    return pd.DataFrame()

                df = pd.DataFrame({
                    "Date": pd.to_datetime(timestamps, unit="s").date,
                    "Price": close_prices,
                }).dropna()

                df = df.set_index("Date")
                df["Price"] = df["Price"].ffill().bfill()
                df["Returns"] = df["Price"].pct_change()
                df = df.dropna()

                if not df.empty:
                    return df

            elif response.status_code == 429:
                logger.warning(f"Direct API 429 Rate Limit for {ticker} on attempt {attempt}.")

        except Exception as e:
            logger.warning(f"Direct API fetch error for {ticker} on attempt {attempt}: {e}")

        time.sleep(2 * attempt)

    return pd.DataFrame()


def extract_data(
    tickers: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> dict[str, pd.DataFrame]:
    """Extract historical stock data for multiple tickers using direct v8 API calls."""
    all_stock_data: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        logger.info(f"Extracting historical data for {ticker}...")
        df = _fetch_ticker_direct_api(ticker, start_date, end_date)

        if not df.empty:
            all_stock_data[ticker] = df
            logger.info(f"Successfully fetched data for {ticker} ({len(df)} rows).")
        else:
            logger.error(f"Failed to fetch market data for {ticker}.")

        time.sleep(1)  # Gentle delay between requests

    return all_stock_data