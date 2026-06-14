from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


st.set_page_config(
    page_title="Smart Food Waste Reduction AI Agent",
    page_icon="🥬",
    layout="wide",
)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def uploaded_or_sample(uploaded_file, sample_path: Path) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return load_csv(sample_path)


def prepare_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    prepared = inventory.copy()
    prepared["expiry_date"] = pd.to_datetime(prepared["expiry_date"])
    prepared["days_to_expiry"] = (prepared["expiry_date"].dt.date - date.today()).apply(
        lambda delta: delta.days
    )
    prepared["stock_value"] = prepared["current_stock"] * prepared["unit_cost"]
    return prepared


def waste_risk_score(inventory: pd.DataFrame) -> int:
    if inventory.empty:
        return 0
    expiring_soon = (inventory["days_to_expiry"] <= 7).mean()
    overstock = (
        inventory["current_stock"]
        / inventory["daily_sales_avg"].replace(0, 1)
        > inventory["shelf_life_days"]
    ).mean()
    return int(round((0.65 * expiring_soon + 0.35 * overstock) * 100))


def render_metric_cards(inventory: pd.DataFrame) -> None:
    total_products = inventory["product_id"].nunique()
    inventory_value = inventory["stock_value"].sum()
    expiring_products = int((inventory["days_to_expiry"] <= 7).sum())
    risk_score = waste_risk_score(inventory)

    cols = st.columns(4)
    cols[0].metric("Total Products", f"{total_products:,}")
    cols[1].metric("Inventory Value", f"${inventory_value:,.0f}")
    cols[2].metric("Expiring Products", f"{expiring_products:,}")
    cols[3].metric("Waste Risk Score", f"{risk_score}/100")


def main() -> None:
    st.title("Smart Food Waste Reduction AI Agent")
    st.caption("Hackathon dashboard for supermarket inventory, expiry, and waste risk.")

    with st.sidebar:
        st.header("Data Uploads")
        sales_file = st.file_uploader("Upload sales_data.csv", type=["csv"])
        inventory_file = st.file_uploader("Upload inventory.csv", type=["csv"])
        st.info("Using bundled sample datasets until custom CSVs are uploaded.")

    sales = uploaded_or_sample(sales_file, DATA_DIR / "sales_data.csv")
    inventory = prepare_inventory(uploaded_or_sample(inventory_file, DATA_DIR / "inventory.csv"))

    render_metric_cards(inventory)

    st.subheader("Dashboard Home")
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("#### Inventory Overview")
        st.dataframe(
            inventory[
                [
                    "product_name",
                    "category",
                    "current_stock",
                    "unit_cost",
                    "expiry_date",
                    "days_to_expiry",
                    "stock_value",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.markdown("#### Inventory Value by Category")
        value_by_category = (
            inventory.groupby("category", as_index=False)["stock_value"].sum().sort_values(
                "stock_value", ascending=False
            )
        )
        st.plotly_chart(
            px.bar(
                value_by_category,
                x="category",
                y="stock_value",
                color="category",
                labels={"stock_value": "Inventory Value", "category": "Category"},
            ),
            use_container_width=True,
        )

    st.markdown("#### Sales Data Preview")
    st.dataframe(sales.head(100), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
