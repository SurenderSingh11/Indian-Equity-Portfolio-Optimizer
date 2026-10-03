"""Quantitative Backtesting Engine for Portfolio Optimization."""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple

from .extractor import extract_data
from .model import generate_forecasts
from .optimizer import optimize_portfolio
from .database import get_db_engine, save_dataframe

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Rolling-window backtesting engine comparing Prophet-driven portfolio against benchmark."""

    def __init__(
        self,
        tickers: list[str],
        benchmark_ticker: str = "^NSEI",
        risk_free_rate: float = 0.06,  # 6% RBI repo rate baseline
        rebalance_freq_days: int = 21,  # ~1 trading month
        training_window_days: int = 252 * 2,  # 2-year lookback training window
    ):
        self.tickers = tickers
        self.benchmark_ticker = benchmark_ticker
        self.risk_free_rate = risk_free_rate
        self.rebalance_freq_days = rebalance_freq_days
        self.training_window_days = training_window_days

    def _fetch_and_prepare_data(
        self, start_date: str, end_date: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Fetch historical prices for assets and benchmark using updated batch extractor."""
        all_tickers = list(set(self.tickers + [self.benchmark_ticker]))
        extracted_data = extract_data(all_tickers, start_date=start_date, end_date=end_date)

        if self.benchmark_ticker not in extracted_data:
            raise ValueError(f"Failed to fetch benchmark data for {self.benchmark_ticker}")

        # Extract asset prices
        price_dict = {
            t: extracted_data[t]["Price"] for t in self.tickers if t in extracted_data
        }
        if not price_dict:
            raise ValueError("No asset ticker data retrieved from batch extractor.")

        asset_prices = pd.DataFrame(price_dict).ffill().bfill()

        # Extract benchmark prices aligned with asset dates
        benchmark_prices = (
            extracted_data[self.benchmark_ticker]["Price"]
            .reindex(asset_prices.index)
            .ffill()
            .bfill()
        )

        return asset_prices, benchmark_prices

    def run_backtest(
        self, start_date: str, end_date: str
    ) -> Tuple[pd.DataFrame, dict[str, float]]:
        """Run rolling-window rebalancing simulation over historical range."""
        asset_prices, benchmark_prices = self._fetch_and_prepare_data(start_date, end_date)
        dates = asset_prices.index

        if len(dates) < self.training_window_days + self.rebalance_freq_days:
            raise ValueError("Insufficient data range for training window and rebalance frequency.")

        daily_returns = asset_prices.pct_change().dropna()
        benchmark_returns = benchmark_prices.pct_change().dropna()

        portfolio_daily_returns = []
        portfolio_dates = []

        # Default equal weighting across active tickers
        current_weights = np.ones(len(self.tickers)) / len(self.tickers)

        # Rolling window simulation
        for i in range(self.training_window_days, len(dates), self.rebalance_freq_days):
            train_prices = asset_prices.iloc[i - self.training_window_days : i]

            # 1. Generate Prophet forecasts
            forecasts = {}
            for ticker in self.tickers:
                if ticker not in train_prices.columns:
                    continue
                ticker_df = pd.DataFrame(
                    {
                        "Date": train_prices.index,
                        "Price": train_prices[ticker].values,
                        "Returns": train_prices[ticker].pct_change().fillna(0).values,
                    }
                ).set_index("Date")

                try:
                    forecast = generate_forecasts(ticker_df, days_ahead=self.rebalance_freq_days)
                    forecasts[ticker] = forecast
                except Exception as e:
                    logger.warning(f"Forecast failed for {ticker} at index {i}: {e}")

            # 2. Optimize portfolio weights
            if forecasts:
                try:
                    optimized_weights = optimize_portfolio(forecasts, train_prices)
                    current_weights = np.array(
                        [optimized_weights.get(t, 0.0) for t in self.tickers]
                    )
                except Exception as e:
                    logger.warning(f"Optimization failed at index {i}, retaining existing weights: {e}")

            # 3. Apply weights to test window
            test_start = i
            test_end = min(i + self.rebalance_freq_days, len(dates))
            test_returns = daily_returns.iloc[test_start:test_end]

            period_portfolio_returns = test_returns.dot(current_weights)
            portfolio_daily_returns.extend(period_portfolio_returns.values)
            portfolio_dates.extend(test_returns.index)

        # Build equity curves
        port_ret_series = pd.Series(
            portfolio_daily_returns, index=portfolio_dates, name="Portfolio_Returns"
        )
        bench_ret_series = (
            benchmark_returns.reindex(portfolio_dates).fillna(0).rename("Benchmark_Returns")
        )

        equity_df = pd.DataFrame(
            {
                "Portfolio_Value": (1 + port_ret_series).cumprod(),
                "Benchmark_Value": (1 + bench_ret_series).cumprod(),
                "Portfolio_Returns": port_ret_series,
                "Benchmark_Returns": bench_ret_series,
            }
        )

        metrics = self.calculate_metrics(port_ret_series, bench_ret_series)
        return equity_df, metrics

    def calculate_metrics(
        self, port_returns: pd.Series, bench_returns: pd.Series
    ) -> dict[str, float]:
        """Compute performance and risk metrics."""
        trading_days = 252

        cum_port_return = (1 + port_returns).prod() - 1
        cum_bench_return = (1 + bench_returns).prod() - 1

        ann_port_return = (1 + cum_port_return) ** (trading_days / len(port_returns)) - 1
        ann_bench_return = (1 + cum_bench_return) ** (trading_days / len(bench_returns)) - 1

        ann_volatility = port_returns.std() * np.sqrt(trading_days)

        excess_returns = port_returns - (self.risk_free_rate / trading_days)
        sharpe_ratio = (excess_returns.mean() * trading_days) / (ann_volatility + 1e-9)

        downside_returns = port_returns[port_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(trading_days)
        sortino_ratio = (ann_port_return - self.risk_free_rate) / (downside_std + 1e-9)

        cum_equity = (1 + port_returns).cumprod()
        peak = cum_equity.cummax()
        drawdown = (cum_equity - peak) / peak
        max_drawdown = drawdown.min()

        var_95 = np.percentile(port_returns, 5)

        return {
            "cumulative_portfolio_return": float(cum_port_return),
            "cumulative_benchmark_return": float(cum_bench_return),
            "annualized_portfolio_return": float(ann_port_return),
            "annualized_benchmark_return": float(ann_bench_return),
            "annualized_volatility": float(ann_volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "max_drawdown": float(max_drawdown),
            "var_95": float(var_95),
        }

    def save_results_to_db(
        self,
        equity_df: pd.DataFrame,
        metrics: dict[str, float],
        schema: str = "dev",
    ) -> None:
        """Persist performance curves and risk metrics to Supabase."""
        engine = get_db_engine()

        curve_df = equity_df.reset_index().rename(columns={"index": "date"})
        curve_df["created_at"] = datetime.utcnow()

        metrics_df = pd.DataFrame([metrics])
        metrics_df["created_at"] = datetime.utcnow()

        save_dataframe(curve_df, table_name="backtest_equity_curve", engine=engine, schema=schema)
        save_dataframe(metrics_df, table_name="backtest_metrics", engine=engine, schema=schema)
        logger.info(f"Successfully persisted backtest output to schema '{schema}'.")


def run_backtest_pipeline(
    tickers: list[str],
    start_date: str,
    end_date: str,
    schema: str = "dev",
) -> None:
    """Execution wrapper."""
    logger.info(f"Starting backtesting engine ({start_date} to {end_date}) on schema '{schema}'...")
    engine = BacktestEngine(tickers=tickers)
    equity_df, metrics = engine.run_backtest(start_date=start_date, end_date=end_date)

    logger.info("Backtest Metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    engine.save_results_to_db(equity_df, metrics, schema=schema)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    run_backtest_pipeline(
        tickers=sample_tickers,
        start_date="2022-01-01",
        end_date="2025-01-01",
        schema="dev",
    )