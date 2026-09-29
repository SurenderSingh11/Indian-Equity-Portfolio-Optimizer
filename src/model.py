"""Prophet model for one-step forward prediction."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
from prophet import Prophet

logger = logging.getLogger(__name__)


def _get_us_trading_holidays(start_year: int = 2020, end_year: int = 2030) -> pd.DataFrame:
    """
    Generate US market trading holidays.

    Includes major holidays when US stock markets are closed:
    - New Year's Day (Jan 1, or observed)
    - Martin Luther King Jr. Day (3rd Monday in January)
    - Presidents Day (3rd Monday in February)
    - Good Friday (varies, but markets close early or close)
    - Memorial Day (last Monday in May)
    - Juneteenth (June 19, or observed)
    - Independence Day (July 4, or observed)
    - Labor Day (1st Monday in September)
    - Thanksgiving (4th Thursday in November)
    - Christmas (December 25, or observed)

    Args:
        start_year: Start year for holiday list
        end_year: End year for holiday list

    Returns:
        DataFrame with columns: holiday, ds, lower_window, upper_window
    """
    holidays_list = []

    for year in range(start_year, end_year + 1):
        # New Year's Day (January 1, or observed if on weekend)
        new_years = date(year, 1, 1)
        if new_years.weekday() == 5:  # Saturday -> Friday before
            new_years = new_years - timedelta(days=1)
        elif new_years.weekday() == 6:  # Sunday -> Monday after
            new_years = new_years + timedelta(days=1)
        holidays_list.append({"holiday": "new_years", "ds": new_years})

        # Martin Luther King Jr. Day (3rd Monday in January)
        # First find the first Monday in January, then add 14 days
        jan_1 = date(year, 1, 1)
        first_monday = jan_1 + timedelta(days=(7 - jan_1.weekday()) % 7)
        mlk_day = first_monday + timedelta(days=14)  # 3rd Monday
        holidays_list.append({"holiday": "mlk_day", "ds": mlk_day})

        # Presidents Day (3rd Monday in February)
        feb_1 = date(year, 2, 1)
        first_monday = feb_1 + timedelta(days=(7 - feb_1.weekday()) % 7)
        presidents_day = first_monday + timedelta(days=14)  # 3rd Monday
        holidays_list.append({"holiday": "presidents_day", "ds": presidents_day})

        # Good Friday calculation (Friday before Easter)
        # Easter calculation using Gregorian algorithm (anonymous algorithm)
        # More readable variable names
        century = year // 100
        year_in_century = year % 100
        golden_number = year % 19

        century_leap = century // 4
        century_remainder = century % 4
        correction = (century + 8) // 25
        century_adjust = (century - correction + 1) // 3
        epact = (19 * golden_number + century - century_leap - century_adjust + 15) % 30

        year_leap = year_in_century // 4
        year_remainder = year_in_century % 4
        day_of_week = (32 + 2 * century_remainder + 2 * year_leap - epact - year_remainder) % 7

        easter_correction = (golden_number + 11 * epact + 22 * day_of_week) // 451
        easter_month = (epact + day_of_week - 7 * easter_correction + 114) // 31
        easter_day = ((epact + day_of_week - 7 * easter_correction + 114) % 31) + 1

        easter = date(year, easter_month, easter_day)
        good_friday = easter - timedelta(days=2)  # Friday before Easter Sunday
        holidays_list.append({"holiday": "good_friday", "ds": good_friday})

        # Memorial Day (last Monday in May)
        may_31 = date(year, 5, 31)
        memorial_day = may_31 - timedelta(days=may_31.weekday())
        holidays_list.append({"holiday": "memorial_day", "ds": memorial_day})

        # Juneteenth (June 19, or observed if on weekend)
        juneteenth = date(year, 6, 19)
        if juneteenth.weekday() == 5:  # Saturday -> Friday before
            juneteenth = juneteenth - timedelta(days=1)
        elif juneteenth.weekday() == 6:  # Sunday -> Monday after
            juneteenth = juneteenth + timedelta(days=1)
        holidays_list.append({"holiday": "juneteenth", "ds": juneteenth})

        # Independence Day (July 4, or observed if on weekend)
        july_4 = date(year, 7, 4)
        if july_4.weekday() == 5:  # Saturday -> Friday before
            july_4 = july_4 - timedelta(days=1)
        elif july_4.weekday() == 6:  # Sunday -> Monday after
            july_4 = july_4 + timedelta(days=1)
        holidays_list.append({"holiday": "independence_day", "ds": july_4})

        # Labor Day (1st Monday in September)
        sep_1 = date(year, 9, 1)
        labor_day = sep_1 + timedelta(days=(7 - sep_1.weekday()) % 7)
        holidays_list.append({"holiday": "labor_day", "ds": labor_day})

        # Thanksgiving (4th Thursday in November)
        nov_1 = date(year, 11, 1)
        first_thursday = nov_1 + timedelta(days=(3 - nov_1.weekday()) % 7)
        thanksgiving = first_thursday + timedelta(days=21)  # 4th Thursday
        holidays_list.append({"holiday": "thanksgiving", "ds": thanksgiving})

        # Christmas (December 25, or observed if on weekend)
        christmas = date(year, 12, 25)
        if christmas.weekday() == 5:  # Saturday -> Friday before
            christmas = christmas - timedelta(days=1)
        elif christmas.weekday() == 6:  # Sunday -> Monday after
            christmas = christmas + timedelta(days=1)
        holidays_list.append({"holiday": "christmas", "ds": christmas})

    # Create DataFrame with Prophet holiday format
    holidays_df = pd.DataFrame(holidays_list)
    holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])
    holidays_df["lower_window"] = -1  # Day before holiday can also be affected
    holidays_df["upper_window"] = 1  # Day after holiday can also be affected

    return holidays_df


class ProphetModel:
    """Prophet model for forecasting stock prices."""

    def __init__(self) -> None:
        """Initialise Prophet model."""
        self.model: Prophet | None = None

    def fit(self, price_series: pd.Series) -> ProphetModel:
        """
        Fit Prophet model to price series with trading holidays.

        Args:
            price_series: Historical price series with datetime index

        Returns:
            Self (ProphetModel instance) for method chaining
        """
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        df = pd.DataFrame({"ds": price_series.index, "y": price_series.values})

        # Get trading holidays for the date range in the data
        start_date = price_series.index.min()
        end_date = price_series.index.max()

        if isinstance(start_date, date):
            start_year = start_date.year
            end_year = end_date.year
        else:
            start_year = pd.to_datetime(start_date).year
            end_year = pd.to_datetime(end_date).year

        # Get holidays and filter to relevant date range
        holidays = _get_us_trading_holidays(start_year - 1, end_year + 1)
        holidays = holidays[
            (holidays["ds"] >= pd.to_datetime(start_date))
            & (holidays["ds"] <= pd.to_datetime(end_date))
        ]

        # Initialise Prophet with holidays and seasonality
        # For stocks, we want to explicitly configure seasonalities
        prophet_params = {
            "yearly_seasonality": True,  # Year-end, January effects
            "weekly_seasonality": True,  # Day-of-week effects (Monday/Friday patterns)
            "daily_seasonality": False,  # Not relevant for daily closing prices
            "seasonality_mode": "additive",  # Additive seasonality for stocks
            "seasonality_prior_scale": 10.0,  # Strong seasonality signals
        }

        if not holidays.empty:
            prophet_params["holidays"] = holidays
            logger.info(f"Using {len(holidays)} trading holidays for Prophet model")
        else:
            logger.warning("No holidays found for date range, using Prophet without holidays")

        self.model = Prophet(**prophet_params)
        self.model.fit(df)

        return self

    def predict_next(self, price_series: pd.Series) -> float:
        """
        Fit model and predict next day's price in one step.

        Args:
            price_series: Historical price series including current day

        Returns:
            Predicted price for next day
        """

        self.fit(price_series)

        # Get the last date from the series
        last_date = price_series.index[-1]

        # Create future dataframe with next day
        future = pd.DataFrame({"ds": pd.date_range(start=last_date, periods=2, freq="D")[1:]})

        # Make prediction
        if self.model is None:
            raise RuntimeError("Model not fitted")
        forecast = self.model.predict(future)

        return float(forecast["yhat"].iloc[0])

    def predict_for_tickers(
        self,
        portfolio_data: dict[str, pd.DataFrame],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Predict prices and returns for multiple tickers.

        Args:
            portfolio_data: Dictionary mapping ticker to DataFrame with 'Price' column

        Returns:
            Tuple containing:
            - predictions: dict[str, float] mapping ticker to predicted price
            - predicted_returns: dict[str, float] mapping ticker to predicted return
        """
        predictions: dict[str, float] = {}
        predicted_returns: dict[str, float] = {}

        for ticker in portfolio_data.keys():
            # Get stock data
            df_stock = portfolio_data[ticker]

            # Get current price
            current_price = df_stock["Price"].iloc[-1]

            # Predict next day price
            predicted_price = self.predict_next(df_stock["Price"])
            predictions[ticker] = predicted_price

            # Calculate predicted return
            daily_return = (predicted_price - current_price) / current_price
            predicted_returns[ticker] = daily_return

        return predictions, predicted_returns
