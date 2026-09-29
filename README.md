# Prophet Forecasting for Portfolio Optimisation

An end-to-end machine learning project that forecasts stock and asset prices using Facebook Prophet time series forecasting, then applies Markowitz portfolio optimisation to rebalance portfolios based on these forecasts.

## Project Overview

This project combines machine learning-based time series forecasting with modern portfolio theory to create an automated trading and portfolio management system. The workflow consists of two main components:

1. **Prophet Price Forecasting**: Uses Facebook Prophet to predict future asset prices
2. **Markowitz Portfolio Optimisation**: Optimises portfolio allocation based on forecasted returns and risk

## Components

### 1. Prophet (Time Series Forecasting)

**What is Prophet?**

Prophet is Facebook's open-source time series forecasting tool designed for business forecasting. It handles trends, seasonality, and holidays automatically, making it robust and easy to use for forecasting time series data.

**Why Prophet for Stock Prediction?**

Prophet is excellent for time series forecasting because it:
- Handles trends and seasonality automatically
- Robust to missing data and outliers
- Works well with daily, weekly, and monthly data
- Provides interpretable forecasts with uncertainty intervals
- Requires minimal hyperparameter tuning
- Can incorporate domain knowledge (holidays, events)

**How It Works in This Project:**

- Input: Historical price time series with datetime index
- Model: Prophet fits additive components (trend, seasonality, holidays)
- Output: Forecasted prices for each asset in the portfolio for the next trading day
- Training: The model fits to historical price data and generates one-step-ahead forecasts

### 2. Markowitz Portfolio Optimisation

**What is Markowitz Portfolio Optimisation?**

Markowitz portfolio optimisation, also known as Modern Portfolio Theory (MPT), is a mathematical framework for constructing optimal portfolios. Developed by Harry Markowitz in 1952, it balances the trade-off between expected returns and risk.

**Key Concepts:**

- **Expected Return**: The weighted average of expected returns of individual assets
- **Risk (Volatility)**: Measured as the standard deviation of portfolio returns
- **Correlation**: How assets move relative to each other
- **Efficient Frontier**: The set of optimal portfolios offering the highest expected return for a given level of risk

**The Optimisation Problem:**

The Markowitz model solves:

```
Maximize: μᵀw - λ(wᵀΣw)

Subject to:
- Σwᵢ = 1 (weights sum to 1)
- wᵢ ≥ 0 (long-only portfolio, optional)
- Additional constraints (sector limits, etc.)
```

Where:
- `μ` = vector of expected returns (from Prophet price forecasts)
- `Σ` = covariance matrix of asset returns
- `w` = portfolio weights
- `λ` = risk aversion parameter (configurable in `src/settings.py`)

**How It Works in This Project:**

1. **Input**: Forecasted returns (derived from Prophet price predictions) for each asset
2. **Risk Estimation**: Historical covariance matrix calculated from asset returns
3. **Optimisation**: Solves for optimal weights that maximise risk-adjusted returns using SciPy's SLSQP solver
4. **Output**: Recommended portfolio allocation (weights for each asset)
5. **Rebalancing**: Portfolio is rebalanced based on these optimal weights

## Project Workflow

```
Historical Data Extraction (yfinance)
    ↓
Data Preprocessing & Alignment
    ↓
Prophet Model Training (per ticker)
    ↓
Price Forecasting (next day)
    ↓
Return Calculation (from prices)
    ↓
Mean Returns & Covariance Matrix
    ↓
Markowitz Optimisation (SciPy)
    ↓
Optimal Portfolio Weights
    ↓
Results Logging & Output
```

## Key Features

- **Time Series Forecasting**: Prophet for robust price prediction with trend and seasonality
- **Risk-Aware Optimisation**: Incorporates covariance and correlation for risk management
- **Automated Daily Execution**: GitHub Actions workflow runs daily to generate new recommendations
- **Multi-Asset Support**: Handle portfolios with multiple stocks/assets simultaneously
- **Configurable Parameters**: Easy customisation of risk aversion, minimum allocation, and portfolio tickers
- **Type-Safe Code**: Full type hints and mypy type checking
- **Comprehensive Testing**: Unit tests with pytest and coverage reporting

## Project Structure

```
Prophet-Forecasting-For-Portfolio-Optimisation/
├── README.md
├── pyproject.toml          # Poetry dependencies and project config
├── poetry.lock             # Locked dependency versions
├── Makefile                # Convenience commands (install, test, lint, etc.)
├── Dockerfile              # Docker containerisation (optional)
├── .circleci/
│   └── config.yml          # CircleCI CI/CD configuration
├── .github/
│   └── workflows/
│       └── daily-optimisation.yml  # Daily GitHub Actions workflow
├── src/
│   ├── __init__.py
│   ├── main.py             # Main entry point and run_optimisation()
│   ├── data.py             # Data extraction and preprocessing
│   ├── model.py            # ProphetModel class
│   ├── optimizer.py        # Portfolio optimisation functions
│   └── settings.py          # Configuration constants (tickers, risk params)
├── tests/
│   ├── __init__.py
│   ├── test_data.py        # Tests for data module
│   ├── test_model.py       # Tests for Prophet model
│   └── test_optimizer.py  # Tests for optimisation functions
└── htmlcov/                # Coverage reports (generated)
```

## Installation

### Standard Installation

```bash
# Install dependencies using Poetry
make install-dev

# Or manually
poetry install
```

### Requirements

- Python 3.12+
- Poetry (for dependency management)

The project uses Poetry for dependency management. If you don't have Poetry installed:

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
```

## Usage

### Basic Usage

Run the portfolio optimisation with default settings:

```bash
poetry run python -m src.main
```

Or using the Makefile:

```bash
make run
```

### Configuration

Edit `src/settings.py` to customise:

- **Portfolio Tickers**: Modify `PORTFOLIO_TICKERS` list
- **Risk Aversion**: Adjust `RISK_AVERSION` (higher = more risk averse)
- **Minimum Allocation**: Change `MINIMUM_ALLOCATION` (minimum weight per asset)
- **Date Range**: Update `START_DATE` and `END_DATE` for historical data

Example:

```python
# src/settings.py
PORTFOLIO_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
RISK_AVERSION = 3  # Higher = more risk averse
MINIMUM_ALLOCATION = 0.05  # 5% minimum per asset
START_DATE = "2024-01-01"
```

### Programmatic Usage

```python
from src.main import run_optimisation

# Run optimisation for custom tickers
result = run_optimisation(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2024-01-01",
    end_date="2024-12-31",
    minimum_allocation=0.1  # 10% minimum
)

# Access results
print(f"Date: {result['date']}")
print(f"Optimal Weights: {result['weights']}")
print(f"Predicted Prices: {result['predictions']}")
print(f"Predicted Returns: {result['predicted_returns']}")
```

### Daily Automated Execution

The project includes a GitHub Actions workflow (`.github/workflows/daily-optimisation.yml`) that runs automatically every day at 9:00 AM UTC. Results are saved and can be viewed in the Actions tab.

To manually trigger a run:
1. Go to the repository's Actions tab
2. Select "Daily Portfolio Optimisation"
3. Click "Run workflow"

## Dependencies

### Core Dependencies

- **Machine Learning**: `prophet` - Facebook Prophet for time series forecasting
- **Data Processing**: `pandas`, `numpy` - Data manipulation and numerical operations
- **Optimisation**: `scipy` - Portfolio optimisation solvers (SLSQP)
- **Data Sources**: `yfinance` - Yahoo Finance API for stock data
- **Visualization**: `matplotlib`, `seaborn` - Results visualization (optional)

### Development Dependencies

- `pytest`, `pytest-cov`, `pytest-mock` - Testing framework
- `black`, `ruff` - Code formatting and linting
- `mypy` - Type checking
- `pre-commit` - Git hooks for code quality

## Development

### Running Tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run specific test file
poetry run pytest tests/test_optimizer.py
```

### Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Type checking
make type-check

# Run all checks (format, lint, type-check, test)
make check
```

### CI/CD

The project uses both CircleCI and GitHub Actions:

- **CircleCI**: Runs on every commit/PR - tests, linting, type checking
- **GitHub Actions**: Runs daily at 9:00 AM UTC - executes portfolio optimisation

View CI status:
- [CircleCI Badge](https://dl.circleci.com/status-badge/img/circleci/XbC7AoPbDq6kv77Q92i6dK/9FPwBXXN1scw4EnPpD9g5m/tree/main.svg?style=svg)

## License

*(To be determined)*

## References

- Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.
- Taylor, S. J., & Letham, B. (2018). Forecasting at Scale. *The American Statistician*, 72(1), 37-45. [Prophet Paper](https://peerj.com/preprints/3190/)

