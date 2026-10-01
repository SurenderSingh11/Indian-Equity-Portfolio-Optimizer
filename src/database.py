"""Database operations for persisting Indian stock market predictions and portfolio allocations to Supabase."""

from __future__ import annotations

import logging
import os
from typing import Any

from supabase import Client, create_client

logger = logging.getLogger(__name__)


def get_supabase_client() -> Client:
    """Create and return Supabase client from environment variables."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        logger.error("Supabase credentials (SUPABASE_URL / SUPABASE_KEY) missing in environment.")
        raise ValueError(
            "Supabase client unavailable. Please set SUPABASE_URL and SUPABASE_KEY environment variables."
        )

    return create_client(url, key)


def save_results_to_supabase(result: dict[str, Any]) -> None:
    """
    Save NSE stock forecasting and portfolio optimization outputs across isolated
    Supabase schemas (dev / prod) and multi-table DDL definitions.

    Args:
        result: Dictionary containing pipeline outputs returned by run_optimisation().
    """
    if not result:
        logger.warning("Empty result dictionary passed to save_results_to_supabase. Skipping DB ingestion.")
        return

    supabase = get_supabase_client()
    target_schema = result.get("schema", "dev")
    as_of_date_str = str(result.get("date"))

    logger.info(f"Initiating DB ingestion into Supabase target schema: [{target_schema}]")

    # --------------------------------------------------------------------------
    # 1. Ingest Forecast Outputs (dev/prod.forecast_outputs)
    # --------------------------------------------------------------------------
    predictions = result.get("predictions", {})
    if predictions:
        forecast_rows = []
        for ticker, pred_price in predictions.items():
            price_val = float(pred_price)
            forecast_rows.append({
                "execution_date": as_of_date_str,
                "ticker": ticker,
                "ds": as_of_date_str,
                "yhat": round(price_val, 4),
                "yhat_lower": round(price_val * 0.95, 4),
                "yhat_upper": round(price_val * 1.05, 4),
                "trend": round(price_val, 4),
                "weekly_seasonality": 0.0,
                "yearly_seasonality": 0.0,
            })

        logger.info(f"Upserting {len(forecast_rows)} rows to {target_schema}.forecast_outputs...")
        supabase.schema(target_schema).table("forecast_outputs").upsert(
            forecast_rows, on_conflict="execution_date,ticker,ds"
        ).execute()

    # --------------------------------------------------------------------------
    # 2. Ingest Model Evaluation Metrics (dev/prod.model_evaluations)
    # --------------------------------------------------------------------------
    metrics = result.get("metrics", {})
    if metrics:
        eval_rows = []
        for ticker, metric_data in metrics.items():
            eval_rows.append({
                "ticker": ticker,
                "horizon_days": 30,
                "mape": float(metric_data.get("mape", 0.0)),
                "rmse": float(metric_data.get("rmse", 0.0)),
                "mae": float(metric_data.get("mae", 0.0)),
                "coverage_ratio": 0.95,
            })

        logger.info(f"Inserting {len(eval_rows)} evaluation metric rows to {target_schema}.model_evaluations...")
        supabase.schema(target_schema).table("model_evaluations").insert(eval_rows).execute()

    # --------------------------------------------------------------------------
    # 3. Ingest Markowitz Portfolio Allocations (dev/prod.portfolio_allocations)
    # --------------------------------------------------------------------------
    weights = result.get("weights", {})
    if weights:
        allocation_row = {
            "allocation_date": as_of_date_str,
            "strategy_type": "Max_Sharpe_Prophet",
            "weights": {k: float(v) for k, v in weights.items()},
            "expected_return": 0.0,
            "expected_volatility": 0.0,
            "sharpe_ratio": 0.0,
        }

        logger.info(f"Upserting Markowitz portfolio weights to {target_schema}.portfolio_allocations...")
        supabase.schema(target_schema).table("portfolio_allocations").upsert(
            [allocation_row], on_conflict="allocation_date,strategy_type"
        ).execute()

    # --------------------------------------------------------------------------
    # 4. Ingest Historical Asset Prices (dev/prod.asset_prices)
    # --------------------------------------------------------------------------
    actual_prices_last_month = result.get("actual_prices_last_month", {})
    if actual_prices_last_month:
        price_rows = []
        for ticker, price_series in actual_prices_last_month.items():
            if isinstance(price_series, list):
                for item in price_series:
                    if isinstance(item, dict) and "Date" in item and "Close" in item:
                        price_rows.append({
                            "ticker": ticker,
                            "price_date": str(item["Date"]),
                            "close_price": round(float(item["Close"]), 4),
                            "adj_close": round(float(item.get("Adj Close", item["Close"])), 4),
                        })

        if price_rows:
            logger.info(f"Upserting {len(price_rows)} asset price rows to {target_schema}.asset_prices...")
            supabase.schema(target_schema).table("asset_prices").upsert(
                price_rows, on_conflict="ticker,price_date"
            ).execute()

    logger.info(f"Successfully finished Database Ingestion Pipeline for schema: [{target_schema}]")