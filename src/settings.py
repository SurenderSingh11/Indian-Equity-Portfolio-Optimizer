"""Settings and constants for portfolio optimisation."""
from datetime import datetime

# Risk parameters
MINIMUM_ALLOCATION = 0.01  # Minimum allocation per asset (5%)
MAXIMUM_ALLOCATION = 1
RISK_AVERSION = 3

# Date defaults
START_DATE = "2024-01-01"  # Default start date for historical data
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Stock Allocation
PORTFOLIO_TICKERS = ["AMD", "MSFT", "AAPL", "TSLA", "AMZN", "NVDA"]

# Database
SUPABASE_TABLE_NAME = "stock_optimisation"
