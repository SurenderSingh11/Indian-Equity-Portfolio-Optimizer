"""Prophet model for one-step forward prediction using NSE calendar."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import pandas_market_calendars as mcal
from prophet import Prophet

from .settings import EXCHANGE_CALENDAR, HOLIDAY_NAME_MAP, PROPHET_PARAMS

logger = logging.getLogger(__name__)


def _normalise_holiday_name(name: str) -> str:
    """Convert calendar holiday names into Prophet-friendly labels."""
    if mapped := HOLIDAY_NAME_MAP.get(name):
        return mapped
    cleaned = name.lower()
    for char in ("'", ",", ".", "’"):
        cleaned = cleaned.replace(char, "")
    cleaned = cleaned.replace("&", "and").replace("-", "_")
    cleaned = "_".join(segment for segment in cleaned.split() if segment)
    return cleaned.strip("_")


def _get_indian_trading_holidays(start_year: int = 2020, end_year: int = 2030) -> pd.DataFrame:
    """Fetch Indian (NSE) trading holidays using pandas_market_calendars."""
    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year")

    start = pd.Timestamp(date(start_year, 1, 1))
    end = pd.Timestamp(date(end_year, 12, 31))

    try:
        calendar = mcal.get_calendar(EXCHANGE_CALENDAR)
    except Exception as e:
        logger.warning(f"Could not load calendar '{EXCHANGE_CALENDAR}': {e}. Falling back to default holidays.")
        return pd.DataFrame(columns=["holiday", "ds", "lower_window", "upper_window"])

    holidays: list[dict[str, pd.Timestamp]] = []
    seen: set[tuple[str, pd.Timestamp]] = set()

    if getattr(calendar, "regular_holidays", None) is not None:
        for rule in calendar.regular_holidays.rules:
            name = _normalise_holiday_name(getattr(rule, "name", "holiday"))
            for holiday_date in rule.dates(start, end):
                timestamp = pd.Timestamp(holiday_date).normalize()
                if timestamp.tz is not None:
                    timestamp = timestamp.tz_localize(None)
                key = (name, timestamp)
                if key in seen:
                    continue
                seen.add(key)
                holidays.append({"holiday": name, "ds": timestamp})

    if not holidays:
        return pd.DataFrame(columns=["holiday", "ds", "lower_window", "upper_window"])

    holidays_df = pd.DataFrame(holidays).drop_duplicates(subset=["holiday", "ds"])
    holidays_df = holidays_df.sort_values("ds").reset_index(drop=True)
    holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])
    holidays_df["lower_window"] = -1
    holidays_df["upper_window"] = 1

    return holidays_df


class ProphetModel:
    """Prophet model for forecasting stock prices."""

    def __init__(self) -> None:
        self.model: Prophet | None = None

    def fit(self, price_series: pd.Series) -> ProphetModel:
        df = pd.DataFrame({"ds": price_series.index, "y": price_series.values})

        start_date = price_series.index.min()
        end_date = price_series.index.max()

        start_year = start_date.year if isinstance(start_date, date) else pd.to_datetime(start_date).year
        end_year = end_date.year if isinstance(end_date, date) else pd.to_datetime(end_date).year

        holidays = _get_indian_trading_holidays(start_year - 1, end_year + 1)
        holidays = holidays[
            (holidays["ds"] >= pd.to_datetime(start_date))
            & (holidays["ds"] <= pd.to_datetime(end_date))
        ]

        prophet_params = PROPHET_PARAMS.copy()
        if not holidays.empty:
            prophet_params["holidays"] = holidays
            logger.info(f"Using {len(holidays)} NSE trading holidays for Prophet model")
        else:
            logger.warning("No holidays found for date range, running Prophet without holidays")

        self.model = Prophet(**prophet_params)
        self.model.fit(df)
        return self

    def predict_next(self, price_series: pd.Series) -> float:
        self.fit(price_series)
        last_date = price_series.index[-1]
        future = pd.DataFrame({"ds": pd.date_range(start=last_date, periods=2, freq="B")[1:]})

        if self.model is None:
            raise RuntimeError("Model not fitted")
        forecast = self.model.predict(future)
        return float(forecast["yhat"].iloc[0])

    def predict_for_tickers(
        self,
        portfolio_data: dict[str, pd.DataFrame],
    ) -> tuple[dict[str, float], dict[str, float]]:
        predictions: dict[str, float] = {}
        predicted_returns: dict[str, float] = {}

        for ticker, df_stock in portfolio_data.items():
            current_price = df_stock["Price"].iloc[-1]
            predicted_price = self.predict_next(df_stock["Price"])
            
            predictions[ticker] = predicted_price
            predicted_returns[ticker] = (predicted_price - current_price) / current_price

        return predictions, predicted_returns