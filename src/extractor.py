"""Data extraction module for fetching stock data from Yahoo Finance."""

import logging
import time
import pandas as pd
import requests
import yfinance as yf

from .settings import END_DATE, START_DATE

logger = logging.getLogger(__name__)


def _get_yf_session() -> requests.Session:
    """Create a custom HTTP session with browser headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
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

    df["Price"] = df["Price"].ffill().bfill()
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
    """Extract historical stock data for multiple Indian tickers with fallback handling."""
    all_stock_data: dict[str, pd.DataFrame] = {}
    session = _get_yf_session()

    # --- 1. ATTEMPT BATCH DOWNLOAD WITH VALIDATION ---
    batch_df = pd.DataFrame()
    for attempt in range(1, max_retries + 1):
        logger.info(f"Downloading ticker batch (Attempt {attempt}/{max_retries})...")
        try:
            batch_df = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                group_by="ticker",
                progress=False,
                auto_adjust=True,
                session=session,
            )

            # yfinance returns empty DataFrame on 429 rate limit without throwing Exception
            if not batch_df.empty and len(batch_df.columns) > 0:
                # Ensure at least one requested ticker exists in MultiIndex or columns
                if isinstance(batch_df.columns, pd.MultiIndex):
                    has_data = any(t in batch_df.columns.levels[0] for t in tickers)
                else:
                    has_data = True

                if has_data:
                    break

        except Exception as e:
            logger.warning(f"Batch download error on attempt {attempt}: {e}")

        logger.warning(f"Batch attempt {attempt} returned empty data (Rate limited). Backing off...")
        time.sleep(3 * attempt)

    # --- 2. PROCESS BATCH RESULT IF DATA WAS RETRIEVED ---
    if isinstance(batch_df, pd.DataFrame) and not batch_df.empty:
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df = batch_df.copy()
                else:
                    if not isinstance(batch_df.columns, pd.MultiIndex) or ticker not in batch_df.columns.levels[0]:
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
    logger.warning("Batch download failed. Switching to sequential ticker download with delays...")
    for ticker in tickers:
        for attempt in range(1, max_retries + 1):
            try:
                time.sleep(1.5)  # Pause before making individual request
                df = yf.download(
                    tickers=ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                    session=session,
                )

                if not df.empty:
                    df_processed = _process_ticker_dataframe(df)
                    if not df_processed.empty:
                        all_stock_data[ticker] = df_processed
                        logger.info(f"Successfully retrieved data for {ticker} via sequential download.")
                        break

            except Exception as e:
                logger.warning(f"Sequential attempt {attempt} for {ticker} failed: {e}")
                time.sleep(2 * attempt)

    return all_stock_data