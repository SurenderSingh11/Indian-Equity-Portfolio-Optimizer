"""Main entry point for portfolio optimization."""

from typing import Dict, Optional, List
import logging
import pandas as pd

from src.data import extract_data, preprocess_data, append_predictions
from src.model import ProphetModel
from src.optimizer import optimize_portfolio_mean_variance
from src.settings import START_DATE, END_DATE, MINIMUM_ALLOCATION

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_optimization(
    tickers: List[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    minimum_allocation: Optional[float] = None,
) -> Dict:
    """
    Run portfolio optimization: pull data, predict, calculate allocation, and log result.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date for historical data (YYYY-MM-DD)
        end_date: End date for historical data (YYYY-MM-DD)
        minimum_allocation: Minimum allocation per asset. Defaults to MINIMUM_ALLOCATION.

    Returns:
        Dictionary containing optimization results:
        - date: Date optimization was run
        - predictions: Dict of predicted prices
        - current_prices: Dict of current prices
        - predicted_returns: Dict of predicted returns
        - weights: Optimal portfolio weights
    """
    if minimum_allocation is None:
        minimum_allocation = MINIMUM_ALLOCATION

    as_of_date = pd.to_datetime(end_date).date()
    logger.info(f"Starting portfolio optimization for tickers: {tickers} as of {as_of_date}")

    # 1. Extract historical data
    logger.info("Extracting historical data...")
    raw_data = extract_data(tickers, start_date=start_date, end_date=end_date)
    if not raw_data:
        logger.warning("No data extracted. Exiting optimization.")
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

    # 6. Optimize portfolio using predicted returns as expected returns
    logger.info("Calculating optimal portfolio allocation...")
    optimal_weights = optimize_portfolio_mean_variance(
        new_data, minimum_allocation=minimum_allocation
    )

    # 7. Convert weights to dictionary
    weights_dict = optimal_weights.to_dict()

    # 8. Log results
    logger.info("=" * 70)
    logger.info("Portfolio Optimization Results")
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


def main() -> None:
    """Main CLI entry point."""
    tickers = ["KO", "BBVA", "REP.MC", "MSFT", "AAPL"]

    result = run_optimization(tickers=tickers)

    print("\n" + "=" * 70)
    print("Portfolio Optimization Complete")
    print("=" * 70)
    print(f"Date: {result['date']}")
    print(f"Weights: {result['weights']}")


if __name__ == "__main__":
    main()
