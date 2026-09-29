"""Settings and constants for portfolio optimisation (Indian Stock Market)."""
from datetime import datetime

# Risk parameters
MINIMUM_ALLOCATION = 0.05  # Minimum allocation per asset (5%)
MAXIMUM_ALLOCATION = 1.0
RISK_AVERSION = 5.0

# Date defaults
START_DATE = "2024-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Indian Stock Tickers (National Stock Exchange - NSE)
PORTFOLIO_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "SBIN.NS",
    "LT.NS",          
    "WIPRO.NS",   
]

# Database
SUPABASE_TABLE_NAME = "stock_optimisation_store"

# Exchange calendar code for NSE in pandas_market_calendars
EXCHANGE_CALENDAR = "NSE"

# Holiday name mapping for Prophet model
HOLIDAY_NAME_MAP = {
    "Republic Day": "republic_day",
    "Independence Day": "independence_day",
    "Mahatma Gandhi Jayanti": "gandhi_jayanti",
    "Diwali": "diwali",
    "Holi": "holi",
    "Good Friday": "good_friday",
    "Christmas": "christmas",
}

# Prophet model parameters
PROPHET_PARAMS = {
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
}