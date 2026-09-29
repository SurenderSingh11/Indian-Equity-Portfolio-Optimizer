"""Main entry point for portfolio optimisation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import append_predictions, extract_data, preprocess_data
from src.model import ProphetModel
from src.optimizer import optimize_portfolio_mean_variance
from src.settings import END_DATE, PORTFOLIO_TICKERS, START_DATE

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_optimisation(
    tickers: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> dict[str, Any]:
    """
    Run portfolio optimisation: pull data, predict, calculate allocation, and log result.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date for historical data (YYYY-MM-DD format). Defaults to START_DATE.
        end_date: End date for historical data (YYYY-MM-DD format). Defaults to END_DATE.
        minimum_allocation: Minimum allocation per asset. Defaults to MINIMUM_ALLOCATION if None.

    Returns:
        Dictionary containing optimisation results with keys:
        - date: date object representing date optimisation was run
        - predictions: dict[str, float] of predicted prices for each ticker
        - current_prices: dict[str, float] of current prices for each ticker
        - predicted_returns: dict[str, float] of predicted returns for each ticker
        - weights: dict[str, float] of optimal portfolio weights for each ticker

    Returns empty dict if data extraction fails.
    """

    as_of_date = pd.to_datetime(end_date).date()
    logger.info(f"Starting portfolio optimisation for tickers: {tickers} as of {as_of_date}")

    # 1. Extract historical data
    logger.info("Extracting historical data...")
    raw_data = extract_data(tickers, start_date=start_date, end_date=end_date)
    if not raw_data:
        logger.warning("No data extracted. Exiting optimisation.")
        return {}

    # 2. Preprocess historical data
    logger.info("Preprocessing data...")
    portfolio_data = preprocess_data(raw_data)

    # 3. Predict next step using Prophet
    logger.info("Generating predictions...")
    model = ProphetModel()
    predictions, predicted_returns = model.predict_for_tickers(portfolio_data)

    # 4. Append predictions to historical data
    new_data = append_predictions(portfolio_data, predictions, predicted_returns)

    # 5. Current prices for logging
    current_prices = {ticker: df["Price"].iloc[-1] for ticker, df in portfolio_data.items()}

    # 6. Optimise portfolio using predicted returns as expected returns
    logger.info("Calculating optimal portfolio allocation...")
    optimal_weights = optimize_portfolio_mean_variance(new_data)

    # 7. Convert weights to dictionary
    weights_dict = optimal_weights.to_dict()

    # 8. Log results
    logger.info("=" * 70)
    logger.info("Portfolio Optimisation Results")
    logger.info("=" * 70)
    logger.info(f"Date: {as_of_date}")

    logger.info("\nPredicted Prices (Next Day):")
    for ticker, price in predictions.items():
        logger.info(f"  {ticker}: ${price:.2f} (Current: ${current_prices[ticker]:.2f})")

    logger.info("\nPredicted Returns:")
    for ticker, ret in predicted_returns.items():
        logger.info(f"  {ticker}: {ret*100:.2f}%")

    logger.info("\nOptimal Portfolio Weights:")
    for ticker, weight in weights_dict.items():
        logger.info(f"  {ticker}: {weight*100:.2f}%")

    return {
        "date": as_of_date,
        "predictions": predictions,
        "current_prices": current_prices,
        "predicted_returns": predicted_returns,
        "weights": weights_dict,
    }


def save_results_to_files(
    result: dict[str, Any], results_dir: str | Path = "results"
) -> tuple[Path, Path]:
    """
    Save optimisation results to text and JSON files.

    Args:
        result: Dictionary containing optimisation results from run_optimisation()
        results_dir: Directory to save results files (default: "results")

    Returns:
        Tuple of (txt_file_path, json_file_path)
    """
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)

    today = date.today().isoformat()

    # Save text summary
    txt_file = results_path / f"optimisation-{today}.txt"
    with txt_file.open("w") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("Portfolio Optimisation Complete\n")
        f.write("=" * 70 + "\n")
        f.write(f"Date: {result.get('date', 'N/A')}\n")
        f.write(f"Weights: {result.get('weights', {})}\n")

    # Convert date to ISO string for JSON serialization
    if "date" in result and hasattr(result["date"], "isoformat"):
        result_copy = result.copy()
        result_copy["date"] = result["date"].isoformat()
    else:
        result_copy = result

    # Save JSON file
    json_file = results_path / f"optimisation-{today}.json"
    with json_file.open("w") as f:
        json.dump(result_copy, f, indent=2, default=str)

    return txt_file, json_file


def main(save_to_files: bool = False, results_dir: str | Path = "results") -> None:
    """
    Main CLI entry point.

    Args:
        save_to_files: If True, save results to files in addition to printing
        results_dir: Directory to save results files (only used if save_to_files=True)
    """
    try:
        result = run_optimisation(tickers=PORTFOLIO_TICKERS)

        # Print to console
        print("\n" + "=" * 70)
        print("Portfolio Optimisation Complete")
        print("=" * 70)
        print(f"Date: {result['date']}")
        print(f"Weights: {result['weights']}")

        # Save to files if requested
        if save_to_files:
            txt_file, json_file = save_results_to_files(result, results_dir)
            print(f"\nResults saved to {txt_file} and {json_file}")

    except Exception as e:
        print(f"Error during optimisation: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run portfolio optimisation and optionally save results to files"
    )
    parser.add_argument(
        "--save-to-files",
        action="store_true",
        help="Save results to text and JSON files in the results directory",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results files (default: results)",
    )
    args = parser.parse_args()

    main(save_to_files=args.save_to_files, results_dir=args.results_dir)
