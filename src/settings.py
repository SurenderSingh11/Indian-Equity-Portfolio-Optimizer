"""Settings and constants for portfolio optimisation."""
from datetime import datetime

# Risk parameters
MINIMUM_ALLOCATION = 0.05  # Minimum allocation per asset (5%)
RISK_AVERSION = 3

# Date defaults
START_DATE = "2024-01-01"  # Default start date for historical data
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Stock Allocation
PORTFOLIO_TICKERS = ["KO", "REP.MC", "MSFT", "AAPL", "TSLA", "AMZN"]

# Database
SUPABASE_TABLE_NAME = "stock_optimise"
