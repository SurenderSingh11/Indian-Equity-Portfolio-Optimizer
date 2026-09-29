"""Modernized Streamlit dashboard for Indian Stock Portfolio Forecasts with Entra ID Authentication."""

from __future__ import annotations

import base64
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from streamlit_oauth import OAuth2Component

from src.database import get_supabase_client
from src.settings import SUPABASE_TABLE_NAME

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="NSE Portfolio Allocator | Cloud AI",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# Environment & OAuth configuration
TENANT_ID = os.getenv("TENANT_ID", "ebe591be-cada-4eef-98d8-72b8ce09b40a")
CLIENT_ID = os.getenv("CLIENT_ID", "d04c62a8-1f57-4202-8097-0198b4c385e0")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "VjH8Q~_BSWAj4q4IxgPdu7kbaMPa-RM5Mg9TobpL")

AUTHORIZE_URL = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/authorize"
TOKEN_URL = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
REFRESH_TOKEN_URL = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
REDIRECT_URI = "http://localhost:8501/component/streamlit_oauth.authorize_button"

# Initialize OAuth2 Component
oauth2 = OAuth2Component(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    authorize_endpoint=AUTHORIZE_URL,
    token_endpoint=TOKEN_URL,
    refresh_token_endpoint=REFRESH_TOKEN_URL,
    revoke_token_endpoint=None,
)

# Session state initialization
if "auth_token" not in st.session_state:
    st.session_state["auth_token"] = None


def decode_jwt_payload(token_str: str) -> dict:
    """Helper to decode JWT payload without external libraries."""
    try:
        parts = token_str.split(".")
        if len(parts) >= 2:
            padded = parts[1] + "=" * (-len(parts[1]) % 4)
            decoded_bytes = base64.b64decode(padded)
            return json.loads(decoded_bytes.decode("utf-8"))
    except Exception:
        pass
    return {}


# --- 1. UNAUTHENTICATED VIEW ---
if not st.session_state["auth_token"]:
    st.markdown("<h1 style='text-align: center;'>🔒 Enterprise Portfolio Optimiser</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Prophet ML Forecasting & Markowitz Portfolio Optimisation Engine</p>", unsafe_allow_html=True)
    st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("Please authenticate with your Microsoft Entra ID organizational account to view analytics.")
        result = oauth2.authorize_button(
            name="🔑 Sign in with Microsoft Entra ID",
            redirect_uri=REDIRECT_URI,
            scope="openid profile email https://graph.microsoft.com/User.Read",
            key="entra_id_auth",
        )

        if result and "token" in result:
            st.session_state["auth_token"] = result["token"]
            st.rerun()

    st.stop()


# --- 2. AUTHENTICATED SIDEBAR & USER PROFILE ---
token_data = st.session_state["auth_token"]
id_token_str = token_data.get("id_token", "") if isinstance(token_data, dict) else ""
user_claims = decode_jwt_payload(id_token_str)

user_name = user_claims.get("name", "Authenticated User")
user_email = user_claims.get("preferred_username", user_claims.get("email", "Entra ID Account"))

with st.sidebar:
    st.title("👤 Account Profile")
    st.markdown(f"**{user_name}**")
    st.caption(user_email)
    st.success("Verified via Entra ID")
    st.divider()

    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state["auth_token"] = None
        st.rerun()


# --- 3. SUPABASE DATA RETRIEVAL ---
@st.cache_data(ttl=300)
def load_supabase_predictions() -> pd.DataFrame:
    client = get_supabase_client()
    if client is None:
        return pd.DataFrame()

    response = (
        client.table(SUPABASE_TABLE_NAME)
        .select("*")
        .order("as_of_date", desc=True)
        .order("created_at", desc=True)
        .execute()
    )
    data = getattr(response, "data", None)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if "as_of_date" in df.columns:
        df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])

    df = df.sort_values(["as_of_date", "created_at"], ascending=[True, False])
    df = df.drop_duplicates(subset=["as_of_date", "ticker"], keep="first")

    if "actual_prices_last_month" in df.columns:
        df["actual_prices_last_month"] = df["actual_prices_last_month"].apply(_parse_price_history)

    return df


def _parse_price_history(raw: object) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [float(value) for value in raw]
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, list):
                return [float(value) for value in decoded]
        except json.JSONDecodeError:
            return []
    return []


# --- 4. MAIN DASHBOARD ---
def run_dashboard() -> None:
    st.title("📈 Indian Stock Market Portfolio Allocator")
    st.caption("Meta Prophet Price Predictions & Markowitz Mean-Variance Portfolio Optimisation (NSE)")

    df = load_supabase_predictions()
    if df.empty:
        st.warning("No prediction data found in Supabase. Run 'python -m src.main' to populate initial data.")
        return

    available_dates = sorted(df["as_of_date"].unique(), reverse=True)
    
    # Date Selection Filter
    st.sidebar.divider()
    st.sidebar.subheader("📅 Model Execution Date")
    selected_date = st.sidebar.selectbox(
        "Select Run Date", options=available_dates, format_func=lambda d: d.strftime("%B %d, %Y")
    )

    date_df = df[df["as_of_date"] == selected_date].copy().sort_values("portfolio_weight", ascending=False)

    # Calculate Summary KPI Metrics
    weighted_return = (date_df["predicted_return"] * date_df["portfolio_weight"]).sum() * 100
    top_holding = date_df.iloc[0]
    top_ticker = top_holding["ticker"]
    top_weight = top_holding["portfolio_weight"] * 100

    # Top KPI Display Cards
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Weighted Expected Return", f"{weighted_return:.2f}%", delta=f"{weighted_return:.2f}%")
    kpi2.metric("Top Asset Allocation", f"{top_ticker}", delta=f"{top_weight:.1f}% weight")
    kpi3.metric("Assets Analyzed", f"{len(date_df)} Tickers")

    st.divider()

    # Asset Allocation Visualizations
    st.subheader("Optimal Portfolio Allocations")
    col_chart, col_table = st.columns([1.2, 1])

    with col_chart:
        fig = px.pie(
            date_df,
            names="ticker",
            values="portfolio_weight",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_traces(
            textinfo="label+percent", 
            hovertemplate="<b>%{label}</b><br>Allocation: %{value:.2%}"
        )
        fig.update_layout(
            height=400, 
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        table_df = date_df[["ticker", "predicted_price", "predicted_return", "portfolio_weight"]].copy()
        table_df["predicted_return"] = table_df["predicted_return"] * 100
        table_df["portfolio_weight"] = table_df["portfolio_weight"] * 100
        table_df.columns = ["Ticker", "Predicted Price", "Expected Return", "Allocation"]

        st.dataframe(
            table_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Predicted Price": st.column_config.NumberColumn(format="₹%.2f"),
                "Expected Return": st.column_config.NumberColumn(format="%.2f%%"),
                "Allocation": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

    # Historical Price Trends Expander
    with st.expander("📊 View Historical Price Trends (Last Month)", expanded=False):
        selected_ticker = st.selectbox("Select Ticker for Historical Analysis", options=date_df["ticker"].unique())
        ticker_row = date_df[date_df["ticker"] == selected_ticker].iloc[0]
        prices = ticker_row.get("actual_prices_last_month", [])

        if prices:
            history_fig = go.Figure()
            history_fig.add_trace(go.Scatter(y=prices, mode="lines+markers", name=selected_ticker, line=dict(color="#1f77b4", width=2)))
            history_fig.update_layout(
                title=f"30-Day Historical Closing Prices — {selected_ticker}",
                yaxis_title="Price (INR)",
                xaxis_title="Trading Days",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(history_fig, use_container_width=True)
        else:
            st.info("No historical price array found for this asset.")


if __name__ == "__main__":
    run_dashboard()