"""Data extraction module for fetching stock data from Yahoo Finance."""

import logging
import time
import pandas as pd
import requests
import yfinance as yf

from .settings import END_DATE, START_DATE

logger = logging.getLogger(__name__)


def _get_yf_session() -> requests.Session:
    """Create a custom HTTP session with a standard browser User-Agent to prevent 429 rate limits."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    return session


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
    max_retries: int = 3,
) -> dict[str, pd.DataFrame]:
    """Extract historical stock data for multiple Indian tickers with retry logic and fallback handling."""
    all_stock_data: dict[str, pd.DataFrame] = {}
    session = _get_yf_session()

    # --- 1. ATTEMPT BATCH DOWNLOAD WITH RETRIES ---
    batch_df = pd.DataFrame()
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading ticker batch (Attempt {attempt}/{max_retries})...")
            batch_df = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                group_by="ticker",
                progress=False,
                auto_adjust=True,
                session=session,
            )

            if not batch_df.empty:
                break

        except Exception as e:
            logger.warning(f"Batch download attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)  # Exponential backoff delay

    # --- 2. PROCESS BATCH RESULT IF SUCCESSFUL ---
    if isinstance(batch_df, pd.DataFrame) and not batch_df.empty:
        for ticker in tickers:
            try:
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

        if all_stock_data:
            return all_stock_data

    # --- 3. FALLBACK: SEQUENTIAL SINGLE-TICKER DOWNLOAD ---
    logger.warning("Batch download failed or returned empty data. Switching to sequential ticker download fallback...")
    for ticker in tickers:
        for attempt in range(1, max_retries + 1):
            try:
                ticker_obj = yf.Ticker(ticker, session=session)
                df = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)

                if not df.empty:
                    df_processed = _process_ticker_dataframe(df)
                    if not df_processed.empty:
                        all_stock_data[ticker] = df_processed
                        logger.info(f"Successfully retrieved data for {ticker} via sequential fallback.")
                        break

            except Exception as e:
                logger.warning(f"Sequential download attempt {attempt} for {ticker} failed: {e}")
                time.sleep(1.5 * attempt)

        time.sleep(1)  # Brief pause between sequential requests to avoid rate limits

    return all_stock_data