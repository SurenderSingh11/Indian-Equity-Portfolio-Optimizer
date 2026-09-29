"""Data extraction module for fetching stock data from Yahoo Finance."""

import logging

import pandas as pd
import yfinance as yf

from .settings import END_DATE, START_DATE

logger = logging.getLogger(__name__)


def _process_ticker_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process raw ticker DataFrame: extract close price, calculate returns, and normalise dates."""
    # Keep only Close price column
    if "Close" in df.columns:
        df = df[["Close"]].rename(columns={"Close": "Price"})
    else:
        df = df[["Adj Close"]].rename(columns={"Adj Close": "Price"})

    # Compute daily percentage returns and drop NaN row
    df["Returns"] = df["Price"].pct_change()
    df = df.dropna()

    # Convert index to standard date objects for cross-platform consistency
    df.index = pd.to_datetime(df.index).date
    df.index.name = "Date"

    return df


def _extract_single_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Extract and process historical data for a single ticker."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            logger.warning(f"No price history returned for ticker: {ticker}")
            return None

        df_processed = _process_ticker_dataframe(df)
        return df_processed

    except Exception as e:
        logger.error(f"Error downloading market data for {ticker}: {e}")
        return None


def extract_data(
    tickers: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> dict[str, pd.DataFrame]:
    """Extract historical stock data for multiple Indian tickers."""
    all_stock_data: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        df_processed = _extract_single_ticker_data(ticker, start_date, end_date)
        if df_processed is not None and not df_processed.empty:
            all_stock_data[ticker] = df_processed

    return all_stock_data