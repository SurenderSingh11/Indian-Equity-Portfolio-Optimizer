"""Main entry point for Indian stock market portfolio optimisation ETL and ML pipeline."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

from src.database import save_results_to_supabase
from src.extractor import extract_data
from src.model import ProphetModel
from src.optimiser import optimize_portfolio_mean_variance
from src.processor import append_predictions, collect_recent_prices, preprocess_data
from src.settings import END_DATE, PORTFOLIO_TICKERS, START_DATE

# Configure Enterprise Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("EnterprisePipeline")


def calculate_evaluation_metrics(
    actuals: dict[str, float], predictions: dict[str, float]
) -> dict[str, dict[str, float]]:
    """Calculate MAPE, RMSE, and MAE for predictions against actual prices."""
    metrics = {}
    for ticker in predictions:
        if ticker in actuals and actuals[ticker] > 0:
            y_true = np.array([actuals[ticker]])
            y_pred = np.array([predictions[ticker]])

            mae = float(np.abs(y_true - y_pred)[0])
            rmse = float(np.sqrt((y_true - y_pred) ** 2)[0])
            mape = float((np.abs(y_true - y_pred) / y_true)[0] * 100)

            metrics[ticker] = {
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mape": round(mape, 4),
            }
    return metrics


def run_optimisation(
    tickers: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> dict[str, Any]:
    as_of_date = pd.to_datetime(end_date).date()
    env = os.environ.get("ENV", "development").lower()
    target_schema = "prod" if env == "production" else "dev"

    logger.info(f"Target Environment: [{env.upper()}] | Supabase Schema: [{target_schema}]")
    logger.info(f"Starting portfolio optimisation for NSE tickers: {tickers} as of {as_of_date}")

    logger.info("Extracting historical stock data via Extractor...")
    all_stock_data = extract_data(tickers, start_date=start_date, end_date=end_date)
    if not all_stock_data:
        logger.error("Data extraction returned empty payload. Exiting execution.")
        return {}

    logger.info("Preprocessing market data...")
    portfolio_data = preprocess_data(all_stock_data)

    logger.info("Fitting Meta Prophet models and generating multi-day forecasts...")
    model = ProphetModel()
    predictions, predicted_returns = model.predict_for_tickers(portfolio_data)

    actual_prices_last_month = collect_recent_prices(portfolio_data)
    predicted_data = append_predictions(portfolio_data, predictions, predicted_returns)

    # Compute forecast accuracy metrics
    latest_actuals = {
        ticker: df["y"].iloc[-1]
        for ticker, df in portfolio_data.items()
        if not df.empty and "y" in df.columns
    }
    evaluation_metrics = calculate_evaluation_metrics(latest_actuals, predictions)

    logger.info("Calculating optimal Markowitz portfolio weights (SciPy SLSQP)...")
    weights_dict = optimize_portfolio_mean_variance(predicted_data)

    logger.info("--- OPTIMISATION SUMMARY ---")
    logger.info(f"Execution Date: {as_of_date}")
    for ticker in tickers:
        pred_p = predictions.get(ticker, 0.0)
        weight = weights_dict.get(ticker, 0.0)
        logger.info(f"Asset: {ticker:<12} | Predicted Price: ₹{pred_p:<8.2f} | Weight: {weight * 100:>5.2f}%")

    return {
        "date": as_of_date,
        "schema": target_schema,
        "predictions": predictions,
        "predicted_returns": predicted_returns,
        "actual_prices_last_month": actual_prices_last_month,
        "weights": weights_dict,
        "metrics": evaluation_metrics,
    }


def main() -> None:
    try:
        result = run_optimisation(tickers=PORTFOLIO_TICKERS)
        if not result:
            logger.error("Optimisation job returned empty dictionary.")
            sys.exit(1)

        # Ingest outputs to Supabase schema
        save_results_to_supabase(result)
        logger.info(f"Successfully saved execution outputs to Supabase [{result['schema']}] schema.")

    except Exception as e:
        logger.critical(f"Unhandled exception during pipeline execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()