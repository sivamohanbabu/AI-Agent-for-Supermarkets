from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from agents.forecasting_agent import forecast_product_demand, sales_trend
from agents.inventory_agent import optimize_inventory
from agents.expiry_agent import monitor_expiry
from agents.pricing_agent import apply_dynamic_pricing
from agents.rescue_agent import recommend_food_rescue
from agents.rag_agent import RagAssistant
from agents.supervisor_agent import final_recommendation_report


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
VECTOR_DB_DIR = BASE_DIR / "vector_db"


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

    dashboard_tab, forecast_tab, waste_tab, ai_tab = st.tabs(
        ["Dashboard", "Forecasting & Optimization", "Waste Reduction", "RAG + Supervisor"]
    )

    with dashboard_tab:
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

    with forecast_tab:
        st.subheader("Demand Forecasting Agent")
        product_options = dict(zip(inventory["product_name"], inventory["product_id"]))
        selected_product = st.selectbox("Product", list(product_options.keys()))
        selected_product_id = product_options[selected_product]
        horizon = st.slider("Forecast horizon", min_value=3, max_value=21, value=7)

        forecast = forecast_product_demand(sales, selected_product_id, periods=horizon)
        trend = sales_trend(sales, selected_product_id)

        chart_data = pd.concat(
            [
                trend.rename(columns={"units_sold": "units"}).assign(series="Actual Sales"),
                forecast.forecast.rename(columns={"forecast_units": "units"}).assign(
                    series="Forecast"
                ),
            ],
            ignore_index=True,
        )
        st.plotly_chart(
            px.line(
                chart_data,
                x="date",
                y="units",
                color="series",
                markers=True,
                labels={"units": "Units", "date": "Date"},
            ),
            use_container_width=True,
        )

        current_stock = int(
            inventory.loc[inventory["product_id"] == selected_product_id, "current_stock"].iloc[0]
        )
        reorder_units = max(0, forecast.forecasted_demand - current_stock)
        st.code(
            f"""{forecast.product_name}:
Current Stock: {current_stock}

Forecasted Demand: {forecast.forecasted_demand}

Recommended Reorder:
{reorder_units} Units""",
            language="text",
        )

        st.subheader("Inventory Optimization Agent")
        all_forecasts = {
            product_id: forecast_product_demand(sales, product_id, periods=7).forecasted_demand
            for product_id in inventory["product_id"].unique()
            if product_id in set(sales["product_id"])
        }
        optimization = optimize_inventory(inventory, all_forecasts)
        st.dataframe(optimization, use_container_width=True, hide_index=True)

    with waste_tab:
        st.subheader("AI Waste Reduction System")

        expiry = monitor_expiry(inventory)
        pricing = apply_dynamic_pricing(expiry)
        rescue = recommend_food_rescue(pricing)

        critical_count = int((rescue["expiry_status"] == "Critical").sum())
        warning_count = int((rescue["expiry_status"] == "Warning").sum())
        attention_count = int((rescue["expiry_status"] == "Attention").sum())

        metric_cols = st.columns(3)
        metric_cols[0].metric("Critical: <= 3 Days", critical_count)
        metric_cols[1].metric("Warning: <= 7 Days", warning_count)
        metric_cols[2].metric("Attention: <= 15 Days", attention_count)

        st.markdown("#### Expiry Monitoring Agent")
        st.dataframe(
            rescue[
                [
                    "product_name",
                    "category",
                    "current_stock",
                    "expiry_date",
                    "days_to_expiry",
                    "expiry_status",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with ai_tab:
        st.subheader("RAG Assistant Agent")
        rag = RagAssistant(KNOWLEDGE_BASE_DIR, VECTOR_DB_DIR)
        query = st.text_input("Ask a policy question", value="Why was yogurt given a 30% discount?")
        response = rag.answer(query)
        st.write(response.answer)
        if response.sources:
            st.caption("Sources: " + ", ".join(response.sources))

        st.subheader("Supervisor Agent")
        report = final_recommendation_report(sales, inventory, KNOWLEDGE_BASE_DIR, VECTOR_DB_DIR)
        st.dataframe(report, use_container_width=True, hide_index=True)
        csv = report.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Final Recommendation Report",
            data=csv,
            file_name="final_recommendation_report.csv",
            mime="text/csv",
        )

        st.markdown("#### Dynamic Pricing Agent")
        st.dataframe(
            rescue[
                [
                    "product_name",
                    "selling_price",
                    "recommended_discount_pct",
                    "discounted_price",
                    "expiry_status",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Food Rescue Recommendation Agent")
        st.dataframe(
            rescue[
                [
                    "product_name",
                    "current_stock",
                    "days_to_expiry",
                    "recommended_discount_pct",
                    "food_rescue_action",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
