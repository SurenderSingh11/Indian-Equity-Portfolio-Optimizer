"""Data extraction module for fetching stock data from Yahoo Finance."""

import logging
import requests
import pandas as pd
import yfinance as yf
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

from .settings import END_DATE, START_DATE

logger = logging.getLogger(__name__)


def _get_yf_session() -> requests.Session:
    """Create a custom requests session with browser headers and retries to prevent connection resets."""
    session = requests.Session()
    
    # Modern Chrome User-Agent to pass Yahoo Finance SSL/Header checks
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    
    # Setup automatic retry strategy for connection failures
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    return session


def _process_ticker_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process raw ticker DataFrame: extract close price, calculate returns, and normalise dates."""
    # Handle MultiIndex columns if returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only Close price column
    if "Close" in df.columns:
        df = df[["Close"]].rename(columns={"Close": "Price"})
    elif "Adj Close" in df.columns:
        df = df[["Adj Close"]].rename(columns={"Adj Close": "Price"})
    else:
        raise KeyError("Neither 'Close' nor 'Adj Close' column found in extracted data.")

    # Compute daily percentage returns and drop NaN row
    df["Returns"] = df["Price"].pct_change()
    df = df.dropna()

    # Convert index to standard date objects for cross-platform consistency
    df.index = pd.to_datetime(df.index).date
    df.index.name = "Date"

    return df


def _extract_single_ticker_data(
    ticker: str, start_date: str, end_date: str, session: requests.Session
) -> pd.DataFrame | None:
    """Extract and process historical data for a single ticker."""
    try:
        stock = yf.Ticker(ticker, session=session)
        df = stock.history(start=start_date, end=end_date)

        # Fallback to yf.download if Ticker.history returns empty
        if df.empty:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                session=session,
                progress=False,
            )

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
    session = _get_yf_session()

    for ticker in tickers:
        df_processed = _extract_single_ticker_data(ticker, start_date, end_date, session)
        if df_processed is not None and not df_processed.empty:
            all_stock_data[ticker] = df_processed

    return all_stock_data