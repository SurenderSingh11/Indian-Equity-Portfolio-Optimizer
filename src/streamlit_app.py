"""Streamlit dashboard for viewing Prophet portfolio predictions."""

from __future__ import annotations

import json
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

from src.database import get_supabase_client
from src.settings import SUPABASE_TABLE_NAME

st.set_page_config(page_title="Portfolio Forecast Dashboard", layout="wide")


@st.cache_data(ttl=300)
def load_supabase_predictions() -> pd.DataFrame:
    """Load prediction rows from Supabase, keeping the latest run per date/ticker."""
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
    """Convert JSON payload from Supabase into a list of floats."""
    if raw is None:
        return []

    if isinstance(raw, list):
        return [float(value) for value in raw]

    if isinstance(raw, str):
        try:
            deserialised = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(deserialised, list):
            return [float(value) for value in deserialised]

    return []


def _build_price_history(
    ticker_row: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Create DataFrames for historical and predicted prices to chart."""
    prices = ticker_row.get("actual_prices_last_month", [])
    if not prices:
        return None

    as_of_date: date = ticker_row["as_of_date"]
    n = len(prices)

    actual_index = pd.bdate_range(end=pd.to_datetime(as_of_date), periods=n)
    prediction_date = pd.bdate_range(
        start=pd.to_datetime(as_of_date) + pd.Timedelta(days=1),
        periods=1,
    )[0]

    actual_df = pd.DataFrame(
        {
            "date": actual_index,
            "price": prices,
        }
    )
    predicted_df = pd.DataFrame(
        {
            "date": [prediction_date],
            "price": [ticker_row["predicted_price"]],
        }
    )

    return actual_df, predicted_df


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%" if value is not None else "—"


def main() -> None:
    st.title("📈 Portfolio Forecast Dashboard")
    st.caption(
        "Latest Prophet predictions, portfolio weights, and recent price dynamics from Supabase."
    )

    df = load_supabase_predictions()

    if df.empty:
        st.info("No prediction data available. Run the optimisation pipeline to populate Supabase.")
        return

    available_dates = sorted(df["as_of_date"].unique(), reverse=True)
    selected_date = st.selectbox(
        "Select as-of date",
        options=available_dates,
        format_func=lambda d: d.strftime("%Y-%m-%d"),
    )

    date_df = df[df["as_of_date"] == selected_date].copy()
    date_df = date_df.sort_values("ticker")

    st.subheader("Summary")
    summary_cols = date_df[
        ["ticker", "predicted_price", "predicted_return", "portfolio_weight"]
    ].copy()
    summary_cols["predicted_return_pct"] = summary_cols["predicted_return"].apply(lambda x: x * 100)
    summary_cols = summary_cols.rename(
        columns={
            "ticker": "Ticker",
            "predicted_price": "Predicted Price",
            "predicted_return": "Predicted Return",
            "predicted_return_pct": "Predicted Return (%)",
            "portfolio_weight": "Portfolio Weight",
        }
    )

    st.dataframe(
        summary_cols[["Ticker", "Predicted Price", "Predicted Return (%)", "Portfolio Weight"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Predicted Price": st.column_config.NumberColumn(format="$%.2f"),
            "Predicted Return (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Portfolio Weight": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    weight_chart_df = date_df.set_index("ticker")["portfolio_weight"]
    st.subheader("Portfolio Weights")
    st.bar_chart(weight_chart_df)

    return_chart_df = date_df.set_index("ticker")["predicted_return"] * 100
    st.subheader("Predicted Returns (%)")
    st.bar_chart(return_chart_df)

    tickers = date_df["ticker"].tolist()
    selected_ticker = st.selectbox("Select ticker for detailed view", options=tickers)

    ticker_row = date_df.set_index("ticker").loc[selected_ticker]
    latest_actual = (
        ticker_row["actual_prices_last_month"][-1]
        if ticker_row["actual_prices_last_month"]
        else None
    )

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.metric(
            "Latest Actual Price", f"${latest_actual:.2f}" if latest_actual is not None else "—"
        )
    with metrics_col2:
        st.metric("Predicted Price", f"${ticker_row['predicted_price']:.2f}")
    with metrics_col3:
        st.metric("Predicted Return", _format_percent(ticker_row["predicted_return"]))

    st.subheader(f"Price Trend · {selected_ticker}")
    price_history = _build_price_history(ticker_row)
    if price_history is not None:
        actual_df, predicted_df = price_history
        actual_chart = (
            alt.Chart(actual_df)
            .mark_line(point=True, color="#1f77b4")
            .encode(
                x="date:T",
                y="price:Q",
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("price:Q", format=".2f")],
            )
        )
        predicted_chart = (
            alt.Chart(predicted_df)
            .mark_point(size=100, color="#ff7f0e")
            .encode(
                x="date:T",
                y="price:Q",
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("price:Q", format=".2f")],
            )
        )
        st.altair_chart(actual_chart + predicted_chart, use_container_width=True)
        st.caption(
            "Actual prices over the past month (blue) with the next-day price prediction (orange)."
        )
    else:
        st.info("No historical price data available for this ticker.")


if __name__ == "__main__":
    main()
