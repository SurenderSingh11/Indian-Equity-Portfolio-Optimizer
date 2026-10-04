# 📈 Quantitative Portfolio Optimization Platform

> **Enterprise-oriented quantitative investment platform for automated stock forecasting and portfolio optimization across Indian equities listed on the National Stock Exchange (NSE).**

An end-to-end quantitative platform that combines **time-series forecasting, portfolio optimization, automated data engineering, backtesting, cloud persistence, interactive analytics, and enterprise authentication** into a production-oriented workflow.

The platform uses **Meta Prophet** to generate daily asset forecasts and **Markowitz Mean-Variance Optimization**, solved using **SciPy SLSQP**, to construct constrained, risk-adjusted portfolio allocations.

---

## 🚀 Key Capabilities

- 📊 **Automated NSE market data ingestion** using `yfinance`
- 🔮 **Daily asset price forecasting** using Meta Prophet
- 📐 **Markowitz Mean-Variance Portfolio Optimization**
- ⚙️ **Constrained numerical optimization** using SciPy SLSQP
- 🧪 **Rolling-window portfolio backtesting**
- 📈 **Risk and performance analytics**
- ☁️ **Supabase PostgreSQL persistence**
- 🔄 **Automated daily execution** using GitHub Actions
- 📊 **Interactive Streamlit + Plotly analytics dashboard**
- 🔐 **Microsoft Entra ID SSO authentication**
- 🏗️ **Development and production environment isolation**

---

## 🏛️ System Architecture

The platform follows a modular **data-to-decision pipeline**, where market data flows through preprocessing, forecasting, portfolio optimization, persistence, and visualization.

```text
                        ┌──────────────────────────┐
                        │      GitHub Actions      │
                        │    Scheduled Daily Job   │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────┐
                  │        1. DATA INGESTION           │
                  │  yfinance + NSE Market Calendar    │
                  │    Data Cleaning & Alignment       │
                  └──────────────────┬─────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────┐
                  │      2. PREDICTIVE ANALYTICS       │
                  │            Meta Prophet            │
                  │         Price Forecasting          │
                  │          Expected Returns          │
                  └──────────────────┬─────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────┐
                  │     3. PORTFOLIO OPTIMIZATION      │
                  │       Markowitz Mean-Variance      │
                  │             SciPy SLSQP            │
                  │        Allocation Constraints      │
                  └──────────────────┬─────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────┐
                  │      4. ANALYTICS & STORAGE        │
                  │     Backtesting • Risk Metrics     │
                  │        Supabase PostgreSQL         │
                  └──────────────────┬─────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────┐
                  │       5. ANALYTICS DASHBOARD       │
                  │         Streamlit + Plotly         │
                  │       Microsoft Entra ID SSO       │
                  └────────────────────────────────────┘

---

## 📊 Pipeline Components

### 1. Data Ingestion & Processing

**Files:** `src/extractor.py`, `src/processor.py`

The data layer retrieves historical market data for NSE-listed equities using `yfinance` and prepares a consistent dataset for modelling and portfolio construction.

#### Responsibilities

- Retrieve historical OHLCV market data
- Validate and clean price series
- Align assets across common trading sessions
- Handle NSE trading holidays
- Calculate historical returns
- Prepare model-ready datasets

Example NSE tickers:

```text
RELIANCE.NS
TCS.NS
INFY.NS
HDFCBANK.NS
```

Trading sessions are aligned using:

```text
pandas_market_calendars
```

This ensures that portfolio calculations operate on consistent trading dates across assets.

---

## 🔮 2. Time-Series Forecasting

**File:** `src/model.py`

Each asset is modelled independently using **Meta Prophet** to generate short-term price forecasts.

The underlying additive time-series formulation is:

$$
y(t) = g(t) + s(t) + h(t) + \epsilon_t
$$

Where:

| Component | Description |
|---|---|
| $g(t)$ | Underlying trend component |
| $s(t)$ | Seasonal component |
| $h(t)$ | Holiday effects |
| $\epsilon_t$ | Model error term |

The forecasting pipeline generates a **1-step-ahead predicted price (`yhat`)**.

The predicted price is then transformed into an expected return estimate that becomes an input to the portfolio optimization engine.

---

## 📐 3. Portfolio Optimization

**File:** `src/optimiser.py`

The portfolio construction engine applies **Markowitz Mean-Variance Optimization** to balance expected return against portfolio risk using **SciPy Sequential Least Squares Programming (SLSQP)**.

The optimization objective is:

$$\max_{w} \quad \mu^T w - \lambda(w^T \Sigma w)$$

Subject to the full-investment constraint and allocation bounds:

$$\text{Subject to: } \sum_{i=1}^{N} w_i = 1 \quad \text{and} \quad w_{\min} \leq w_i \leq w_{\max}$$

Where:

| Variable | Description |
|---|---|
| $\mu$ | Vector of predicted asset returns |
| $\Sigma$ | Historical covariance matrix |
| $w$ | Portfolio allocation weights |
| $\lambda$ | Risk-aversion coefficient |

### Optimization Constraints

The engine supports practical portfolio constraints including:

- Full portfolio investment
- Minimum allocation per asset
- Maximum allocation per asset
- Risk-adjusted allocation based on expected returns and covariance

This converts individual asset forecasts into an optimized portfolio allocation rather than treating predictions independently.

---

## 🧪 Backtesting & Risk Analytics

The platform includes a **rolling-window backtesting framework** designed to evaluate how the forecasting and optimization pipeline performs across historical periods.

Portfolio performance is evaluated against the **Nifty 50 Index (`^NSEI`)**.

### Performance Metrics

- **Sharpe Ratio**
- **Sortino Ratio**
- **Maximum Drawdown**
- **95% Value-at-Risk (VaR)**
- **Portfolio Equity Curve**
- **Benchmark Performance**

The configured backtesting setup has produced a historical **Sharpe Ratio above 1.8** during the evaluated period.

> **Note:** Backtesting results are dependent on the selected historical period, model configuration, portfolio constraints, and evaluation assumptions. Historical performance does not guarantee future results.

---

## ☁️ Database & Persistence

**File:** `src/database.py`

The platform persists model outputs and portfolio analytics using **Supabase PostgreSQL**.

### Persisted Data

- Asset price forecasts
- Expected returns
- Portfolio allocation weights
- MAPE
- RMSE
- Backtesting metrics
- Portfolio performance outputs

The application supports separate **development and production environments**, helping isolate development experimentation from production data.

---

## 📊 Interactive Analytics Dashboard

**File:** `src/streamlit_app.py`

The platform provides an interactive **Streamlit dashboard** powered by **Plotly** for visualizing model predictions, portfolio allocations, and historical performance.

### Dashboard Capabilities

- 📈 Historical stock price visualization
- 🔮 Forecasted price trajectories
- 🥧 Portfolio allocation visualization
- 🔥 Covariance / correlation heatmaps
- 📊 Portfolio performance analytics
- 📉 Drawdown analysis
- 🧪 Backtesting results
- 📈 Nifty 50 benchmark comparison

The dashboard provides a single interface for inspecting the outputs of the quantitative pipeline.

---

## 🔐 Enterprise Security

### Microsoft Entra ID SSO

The Streamlit application is protected using **Microsoft Entra ID** authentication.

Authentication is implemented using an **OAuth 2.0 / PKCE-based flow** through `streamlit-oauth`.

### Security Features

- Enterprise identity integration
- Authenticated dashboard access
- OAuth-based authentication
- Corporate user access control
- Separation of application credentials from source code
- Environment-based secret configuration

Sensitive credentials are stored through environment variables and Streamlit secrets rather than being hard-coded into the application.

---

## 🔄 CI/CD & Automation

The daily quantitative pipeline is orchestrated automatically using **GitHub Actions** (`.github/workflows/daily-pipeline.yml`).

- **Automated Trigger:** Executes on a scheduled cron job prior to Indian market open.
- **Pipeline Execution:** Ingests market data, fits Prophet forecasting models, calculates expected returns, and runs SciPy SLSQP optimization.
- **Persistence & Alerting:** Logs portfolio allocations, model forecasts, and risk metrics directly to Supabase PostgreSQL without manual intervention.

---

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.12+ |
| **Package Management** | Poetry |
| **Data Processing** | Pandas, NumPy |
| **Market Data** | yfinance |
| **Trading Calendar** | pandas_market_calendars |
| **Forecasting** | Meta Prophet |
| **Optimization** | SciPy SLSQP |
| **Database** | Supabase PostgreSQL |
| **Dashboard** | Streamlit |
| **Visualization** | Plotly |
| **Authentication** | Microsoft Entra ID / OAuth 2.0 PKCE |
| **CI/CD** | GitHub Actions |
| **Benchmark** | Nifty 50 (`^NSEI`) |

---

## 📁 Project Structure

```text
portfolio-optimiser/
│
├── src/
│   ├── extractor.py          # NSE market data ingestion
│   ├── processor.py          # Data cleaning & preprocessing
│   ├── model.py              # Prophet forecasting models
│   ├── optimiser.py          # Markowitz portfolio optimization
│   ├── database.py           # Supabase PostgreSQL integration
│   ├── main.py               # End-to-end pipeline entry point
│   └── streamlit_app.py      # Interactive analytics dashboard
│
├── .github/
│   └── workflows/
│       └── daily-pipeline.yml
│
├── .streamlit/
│   └── secrets.toml
│
├── pyproject.toml
├── poetry.lock
├── .env
└── README.md
```

---

## 🚀 Installation & Local Setup

### Prerequisites

Before running the project locally, ensure the following are available:

- **Python 3.12+**
- **Poetry**
- **Supabase project**
- **Microsoft Entra ID App Registration**

---

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/portfolio-optimiser.0.1.git

cd portfolio-optimiser.0.1
```

---

### 2. Install Dependencies

Install all project dependencies using Poetry:

```bash
poetry install
```

---

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
SUPABASE_URL="https://your-supabase-project.supabase.co"
SUPABASE_KEY="your-supabase-anon-or-service-role-key"
SUPABASE_TABLE_NAME="portfolio_predictions_dev"
```

> ⚠️ **Never commit `.env`, database credentials, client secrets, or other sensitive configuration files to source control.**

---

### 4. Configure Microsoft Entra ID

Configure Streamlit secrets in:

```text
.streamlit/secrets.toml
```

Example:

```toml
TENANT_ID = "your-azure-tenant-id"
CLIENT_ID = "your-azure-client-id"
CLIENT_SECRET = "your-azure-client-secret"
REDIRECT_URI = "http://localhost:8501/component/streamlit_oauth.authorize_button"
```

---

## 💻 Running the Application

### Run the Quantitative Pipeline

Execute the complete end-to-end pipeline:

```bash
poetry run python -m src.main
```

The pipeline performs:

1. Market data extraction
2. Data validation and processing
3. Forecast generation
4. Expected return calculation
5. Portfolio optimization
6. Risk and performance calculation
7. Database persistence

---

### Launch the Streamlit Dashboard

```bash
poetry run streamlit run src/streamlit_app.py
```

The dashboard will be available through the local URL provided by Streamlit.

---

## 📌 Engineering Highlights

### Automated Data Engineering

Designed an automated Python data pipeline that retrieves NSE equity data, validates and aligns trading sessions, processes historical price data, and prepares datasets for downstream forecasting and portfolio optimization.

### Machine Learning & Forecasting

Implemented individual **Meta Prophet time-series models** to generate short-term asset price forecasts and derive expected returns for portfolio construction.

### Quantitative Portfolio Optimization

Engineered a **Markowitz Mean-Variance Optimization engine** using **SciPy SLSQP** to construct risk-adjusted portfolios under full-investment and asset-level allocation constraints.

### Backtesting & Risk Analytics

Developed a rolling-window backtesting framework to evaluate strategy performance against the **Nifty 50 benchmark**, including Sharpe Ratio, Sortino Ratio, Maximum Drawdown, and Value-at-Risk.

### Production Automation

Automated the complete daily quantitative workflow using **GitHub Actions**, enabling scheduled execution without manual intervention.

### Cloud Data Persistence

Implemented **Supabase PostgreSQL** persistence for forecasts, portfolio allocations, model metrics, and backtesting outputs with development and production environment isolation.

### Enterprise Application Security

Integrated **Microsoft Entra ID SSO** using an **OAuth 2.0 / PKCE-based authentication flow** to secure the Streamlit analytics platform.

---

## ⚠️ Disclaimer

This project is intended for **research, educational, and engineering demonstration purposes only**.

Forecasts and portfolio allocations generated by the system should **not be considered financial advice or recommendations to buy or sell securities**.

Past backtesting performance does not guarantee future results.

---

## 👨‍💻 Author

**Surender Singh**

`Python` • `Machine Learning` • `Quantitative Finance` • `Data Engineering` • `Cloud` • `MLOps`
````
