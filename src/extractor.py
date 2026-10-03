"""Data extraction module for fetching stock data from Yahoo Finance."""

import logging
import time
import pandas as pd
import yfinance as yf

from .settings import END_DATE, START_DATE

logger = logging.getLogger(__name__)


def _process_ticker_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process raw ticker DataFrame: extract close price, calculate returns, and normalise dates."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" in df.columns:
        df = df[["Close"]].rename(columns={"Close": "Price"})
    elif "Adj Close" in df.columns:
        df = df[["Adj Close"]].rename(columns={"Adj Close": "Price"})
    else:
        raise KeyError("Neither 'Close' nor 'Adj Close' column found in extracted data.")

    # Fill occasional missing price points before return calculations
    df["Price"] = df["Price"].ffill().bfill()

    # Compute daily percentage returns and drop initial NaN row
    df["Returns"] = df["Price"].pct_change()
    df = df.dropna()

    df.index = pd.to_datetime(df.index).date
    df.index.name = "Date"

    return df


def extract_data(
    tickers: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> dict[str, pd.DataFrame]:
    """Extract historical stock data for multiple Indian tickers using batch download."""
    all_stock_data: dict[str, pd.DataFrame] = {}

    try:
        # Download all tickers in a single batch request to prevent 429 rate limits
        batch_df = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            group_by="ticker",
            progress=False,
            auto_adjust=True,
        )

        if batch_df.empty:
            logger.warning("Empty data returned for the requested ticker batch.")
            return all_stock_data

        # Process each ticker from the batch result
        for ticker in tickers:
            try:
                # Extract single ticker slice from MultiIndex batch output
                if len(tickers) == 1:
                    df = batch_df.copy()
                else:
                    if ticker not in batch_df.columns.levels[0]:
                        logger.warning(f"Ticker {ticker} not found in downloaded batch data.")
                        continue
                    df = batch_df[ticker].dropna(how="all").copy()

                if df.empty:
                    continue

                df_processed = _process_ticker_dataframe(df)
                if not df_processed.empty:
                    all_stock_data[ticker] = df_processed

            except Exception as e:
                logger.error(f"Error processing market data for {ticker}: {e}")

    except Exception as e:
        logger.error(f"Failed batch download from Yahoo Finance: {e}")

    return all_stock_data