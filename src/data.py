"""Data extraction and preprocessing."""
import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from .settings import END_DATE, START_DATE

logger = logging.getLogger(__name__)


def _process_ticker_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw ticker DataFrame: extract price, calculate returns, normalise dates.

    Args:
        df: Raw DataFrame from yfinance with date index and 'Close' column

    Returns:
        Processed DataFrame with 'Price' and 'Returns' columns and date index
    """
    # Keep only relevant columns
    df = df[["Close"]].rename(columns={"Close": "Price"})

    # Compute daily returns
    df["Returns"] = df["Price"].pct_change()
    df = df.dropna()

    # Convert index to date
    df.index = df.index.date
    df.index.name = "Date"

    return df


def _extract_single_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Extract and process data for a single ticker.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data download (YYYY-MM-DD format)
        end_date: End date for data download (YYYY-MM-DD format)

    Returns:
        Processed DataFrame or None if extraction fails
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            logger.warning(f"No data available for ticker: {ticker}")
            return None

        return _process_ticker_dataframe(df)

    except Exception as e:
        logger.error(f"Error downloading {ticker}: {e}")
        return None


def extract_data(
    tickers: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> dict[str, pd.DataFrame]:
    """
    Extract historical stock data for multiple tickers.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date for data download (YYYY-MM-DD format)
        end_date: End date for data download (YYYY-MM-DD format)

    Returns:
        Dictionary mapping ticker to DataFrame with columns ['Price', 'Returns']
    """
    data_dict: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        processed_df = _extract_single_ticker_data(ticker, start_date, end_date)
        if processed_df is not None:
            data_dict[ticker] = processed_df

    return data_dict


def _normalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise DataFrame index to datetime.date type.

    Args:
        df: DataFrame with date-like index

    Returns:
        DataFrame with normalised date index
    """
    df_copy = df.copy()
    df_copy.index = pd.to_datetime(df_copy.index).date
    return df_copy


def _find_common_dates(data_dict: dict[str, pd.DataFrame]) -> list[date]:
    """
    Find common dates across all tickers in the data dictionary.

    Args:
        data_dict: Dictionary mapping ticker to DataFrame with date index

    Returns:
        Sorted list of common dates
    """
    if not data_dict:
        return []

    date_sets = [set(df.index) for df in data_dict.values()]
    common_dates = set.intersection(*date_sets)
    return sorted(common_dates)


def preprocess_data(data_dict: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Align multiple tickers by common dates while keeping them as a dictionary of DataFrames.

    Args:
        data_dict: Dictionary mapping ticker to DataFrame with date index

    Returns:
        Dictionary of DataFrames where all tickers share the same index (common dates)
    """
    if not data_dict:
        return {}

    # Ensure all indexes are datetime.date
    normalized_dict = {ticker: _normalize_date_index(df) for ticker, df in data_dict.items()}

    # Find intersection of dates across all tickers
    common_dates = _find_common_dates(normalized_dict)

    # Trim each DataFrame to the common dates
    aligned_dict = {ticker: df.loc[common_dates] for ticker, df in normalized_dict.items()}
    return aligned_dict


def _calculate_prediction_date(last_date: date) -> date:
    """
    Calculate the date for the prediction (next day after last date).

    Args:
        last_date: Last date in the historical data

    Returns:
        Prediction date (last_date + 1 day)
    """
    return last_date + timedelta(days=1)


def _create_prediction_row(price: float, returns: float, prediction_date: date) -> pd.DataFrame:
    """
    Create a DataFrame row for prediction data.

    Args:
        price: Predicted price
        returns: Predicted returns
        prediction_date: Date for the prediction

    Returns:
        DataFrame with a single row containing Price and Returns
    """
    return pd.DataFrame(
        {"Price": [price], "Returns": [returns]},
        index=[prediction_date],
    )


def append_predictions(
    portfolio_data: dict[str, pd.DataFrame],
    predictions: dict[str, float],
    predicted_returns: dict[str, float],
) -> dict[str, pd.DataFrame]:
    """
    Append predicted price and return to each ticker's DataFrame.

    Args:
        portfolio_data: Dictionary of historical DataFrames per ticker
        predictions: Dictionary of predicted prices per ticker
        predicted_returns: Dictionary of predicted returns per ticker

    Returns:
        Updated dictionary with an additional row for each ticker
    """
    updated_dict = {}

    for ticker, df in portfolio_data.items():
        df_copy = df.copy()

        last_date = df_copy.index[-1]
        prediction_date = _calculate_prediction_date(last_date)

        # Create and append prediction row
        new_row = _create_prediction_row(
            predictions[ticker], predicted_returns[ticker], prediction_date
        )
        df_copy = pd.concat([df_copy, new_row])

        updated_dict[ticker] = df_copy

    return updated_dict
