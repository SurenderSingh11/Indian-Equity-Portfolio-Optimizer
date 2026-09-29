"""Main entry point for Indian stock market portfolio optimisation."""

from __future__ import annotations

import logging
import sys
from typing import Any

import pandas as pd

from src.database import save_results_to_supabase
from src.extractor import extract_data
from src.model import ProphetModel
from src.optimiser import optimize_portfolio_mean_variance
from src.processor import append_predictions, collect_recent_prices, preprocess_data
from src.settings import END_DATE, PORTFOLIO_TICKERS, START_DATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_optimisation(
    tickers: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> dict[str, Any]:
    as_of_date = pd.to_datetime(end_date).date()
    logger.info(f"Starting portfolio optimisation for NSE tickers: {tickers} as of {as_of_date}")

    logger.info("Extracting historical data from yfinance...")
    all_stock_data = extract_data(tickers, start_date=start_date, end_date=end_date)
    if not all_stock_data:
        logger.warning("No data extracted. Exiting optimisation.")
        return {}

    logger.info("Preprocessing data...")
    portfolio_data = preprocess_data(all_stock_data)

    logger.info("Generating Prophet predictions...")
    model = ProphetModel()
    predictions, predicted_returns = model.predict_for_tickers(portfolio_data)

    actual_prices_last_month = collect_recent_prices(portfolio_data)
    predicted_data = append_predictions(portfolio_data, predictions, predicted_returns)

    logger.info("Calculating optimal Markowitz portfolio allocation...")
    weights_dict = optimize_portfolio_mean_variance(predicted_data)

    logger.info("Portfolio Optimisation Results:")
    logger.info(f"Date: {as_of_date}")

    logger.info("\nPredicted Prices (Next Trading Day):")
    for ticker, price in predictions.items():
        logger.info(f"  {ticker}: ₹{price:.2f}")

    logger.info("\nPredicted Returns:")
    for ticker, ret in predicted_returns.items():
        logger.info(f"  {ticker}: {ret * 100:.2f}%")

    logger.info("\nOptimal Portfolio Weights:")
    for ticker, weight in weights_dict.items():
        logger.info(f"  {ticker}: {weight * 100:.2f}%")

    return {
        "date": as_of_date,
        "predictions": predictions,
        "predicted_returns": predicted_returns,
        "actual_prices_last_month": actual_prices_last_month,
        "weights": weights_dict,
    }


def main() -> None:
    try:
        result = run_optimisation(tickers=PORTFOLIO_TICKERS)
        if not result:
            logger.error("Optimisation returned empty result")
            sys.exit(1)

        save_results_to_supabase(result)
        print("\nResults successfully saved to Supabase database")

    except Exception as e:
        logger.error(f"Error during optimisation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()