"""Data processing module for aligning and prepping stock time series."""

import logging
from datetime import timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def clean_ticker_symbol(ticker: str) -> str:
    """Remove exchange identifiers (.NS, .BO) for clean UI presentation."""
    if not ticker:
        return ""
    return ticker.replace(".NS", "").replace(".BO", "").strip()


def preprocess_data(all_stock_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Align multiple tickers by common trading dates."""
    if not all_stock_data:
        return {}

    normalised_data = {}
    for ticker, df in all_stock_data.items():
        df_copy = df.copy()
        df_copy.index = pd.to_datetime(df_copy.index).date
        normalised_data[ticker] = df_copy

    # Find date intersection across all active tickers
    date_sets = [set(df.index) for df in normalised_data.values()]
    common_dates = sorted(set.intersection(*date_sets))

    aligned_data = {
        ticker: df.loc[common_dates] for ticker, df in normalised_data.items()
    }
    return aligned_data


def append_predictions(
    portfolio_data: dict[str, pd.DataFrame],
    predictions: dict[str, float],
    predicted_returns: dict[str, float],
) -> dict[str, pd.DataFrame]:
    """Append the Prophet predicted next-day price and return row to each ticker."""
    updated_portfolio_data = {}

    for ticker, df in portfolio_data.items():
        df_copy = df.copy()
        last_date = pd.to_datetime(df_copy.index[-1])

        # Get next business day
        prediction_date = pd.bdate_range(start=last_date + timedelta(days=1), periods=1)[0].date()

        new_row = pd.DataFrame(
            {"Price": [predictions[ticker]], "Returns": [predicted_returns[ticker]]},
            index=[prediction_date],
        )
        df_copy = pd.concat([df_copy, new_row])
        updated_portfolio_data[ticker] = df_copy

    return updated_portfolio_data


def collect_recent_prices(
    portfolio_data: dict[str, pd.DataFrame],
    days: int = 30,
) -> dict[str, list[float]]:
    """Collect trailing historical prices for visual display in Streamlit."""
    recent_prices: dict[str, list[float]] = {}

    for ticker, df in portfolio_data.items():
        if df.empty:
            recent_prices[ticker] = []
            continue

        last_date = df.index[-1]
        cutoff = last_date - timedelta(days=days)
        recent_series = df.loc[df.index >= cutoff, "Price"]
        recent_prices[ticker] = [float(value) for value in recent_series.tolist()]

    return recent_prices