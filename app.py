from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from agents.expiry_agent import monitor_expiry
from agents.forecasting_agent import forecast_product_demand, sales_trend
from agents.inventory_agent import optimize_inventory
from agents.pricing_agent import apply_dynamic_pricing
from agents.rag_agent import RagAssistant
from agents.rescue_agent import recommend_food_rescue
from agents.supervisor_agent import final_recommendation_report


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
VECTOR_DB_DIR = BASE_DIR / "vector_db"


st.set_page_config(
    page_title="Smart Food Waste Reduction AI Agent",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f7faf8;
            --surface: #ffffff;
            --ink: #17201d;
            --muted: #52635d;
            --line: #d9e5df;
            --teal: #006b5f;
            --indigo: #3a4a8f;
            --coral: #b8442d;
            --amber: #8a5a00;
            --green-soft: #e7f5ef;
            --blue-soft: #eef3ff;
            --amber-soft: #fff5d6;
            --coral-soft: #fff0ec;
        }
        .stApp {
            background: var(--bg);
            color: var(--ink);
        }
        [data-testid="stSidebar"] {
            background: #10211d;
        }
        [data-testid="stSidebar"] * {
            color: #f4fbf7;
        }
        [data-testid="stSidebar"] .stAlert * {
            color: #17201d;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background: #ffffff;
            border: 1px solid #cfd9d5;
            border-radius: 8px;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p {
            color: #17201d !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
            color: #17201d !important;
            background: #f7faf8;
            border: 1px solid #b8c8c2;
            border-radius: 6px;
        }
        .hero {
            border: 1px solid var(--line);
            background: linear-gradient(135deg, #ffffff 0%, #edf8f4 58%, #f7f1df 100%);
            border-radius: 8px;
            padding: 28px 30px;
            margin: 8px 0 18px;
        }
        .eyebrow {
            color: var(--teal);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .hero h1 {
            font-size: clamp(2rem, 4vw, 3.2rem);
            line-height: 1.05;
            margin: 0 0 10px;
            color: var(--ink);
        }
        .hero p {
            max-width: 860px;
            color: var(--muted);
            font-size: 1.04rem;
            line-height: 1.6;
            margin: 0;
        }
        .section-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin: 22px 0 10px;
        }
        .section-title h2 {
            font-size: 1.22rem;
            margin: 0;
            color: var(--ink);
        }
        .section-title p {
            color: var(--muted);
            margin: 4px 0 0;
            font-size: .94rem;
        }
        .metric-tile {
            min-height: 118px;
            border: 1px solid var(--line);
            background: var(--surface);
            border-radius: 8px;
            padding: 16px 18px;
            box-shadow: 0 8px 22px rgba(23, 32, 29, 0.06);
        }
        .metric-label {
            color: var(--muted);
            font-size: .84rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .metric-value {
            color: var(--ink);
            font-size: 1.9rem;
            font-weight: 850;
            line-height: 1.1;
        }
        .metric-note {
            color: var(--muted);
            font-size: .82rem;
            margin-top: 8px;
        }
        .insight-strip {
            border-left: 5px solid var(--teal);
            background: var(--green-soft);
            padding: 14px 16px;
            border-radius: 6px;
            color: var(--ink);
            margin: 10px 0 18px;
        }
        .status-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }
        .status-box {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px;
            background: var(--surface);
        }
        .status-box strong {
            display: block;
            color: var(--ink);
            font-size: 1.35rem;
        }
        .status-box span {
            color: var(--muted);
            font-size: .9rem;
        }
        .critical { border-top: 4px solid var(--coral); }
        .warning { border-top: 4px solid var(--amber); }
        .attention { border-top: 4px solid var(--indigo); }
        div[data-testid="stMetric"] {
            border: 1px solid var(--line);
            background: var(--surface);
            border-radius: 8px;
            padding: 14px 16px;
        }
        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-weight: 700;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            min-height: 46px;
            border: 1px solid var(--line);
            border-radius: 8px 8px 0 0;
            background: #ffffff;
            color: var(--ink);
            font-weight: 700;
        }
        .stTabs [aria-selected="true"] {
            background: #e7f5ef;
            border-bottom-color: #e7f5ef;
        }
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 6px;
            border: 1px solid var(--teal);
            background: var(--teal);
            color: #ffffff;
            font-weight: 750;
        }
        .stButton > button:focus,
        .stDownloadButton > button:focus,
        input:focus,
        textarea:focus,
        [role="combobox"]:focus {
            outline: 3px solid #9cd4ca !important;
            outline-offset: 2px !important;
        }
        @media (max-width: 900px) {
            .hero { padding: 22px 18px; }
            .status-row { grid-template-columns: 1fr; }
            .metric-value { font-size: 1.55rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
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


def section_title(title: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">
            <div>
                <h2>{title}</h2>
                <p>{caption}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_tile(label: str, value: str, note: str) -> str:
    return f"""
    <div class="metric-tile">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """


def render_metric_cards(inventory: pd.DataFrame) -> None:
    total_products = inventory["product_id"].nunique()
    inventory_value = inventory["stock_value"].sum()
    expiring_products = int((inventory["days_to_expiry"] <= 7).sum())
    risk_score = waste_risk_score(inventory)

    cols = st.columns(4)
    cards = [
        metric_tile("Total Products", f"{total_products:,}", "Unique SKUs in the current file"),
        metric_tile("Inventory Value", f"${inventory_value:,.0f}", "Current stock value at cost"),
        metric_tile("Expiring Products", f"{expiring_products:,}", "Items due within 7 days"),
        metric_tile("Waste Risk Score", f"{risk_score}/100", "Higher score needs faster action"),
    ]
    for col, card in zip(cols, cards):
        col.markdown(card, unsafe_allow_html=True)


def style_plot(fig):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=12, r=12, t=28, b=12),
        legend_title_text="",
        font=dict(family="Arial", size=13, color="#17201d"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#d9e5df")
    return fig


def inventory_columns() -> dict[str, object]:
    return {
        "product_name": st.column_config.TextColumn("Product", help="Product display name"),
        "category": st.column_config.TextColumn("Category", help="Supermarket category"),
        "current_stock": st.column_config.NumberColumn("Stock", help="Current units on hand"),
        "unit_cost": st.column_config.NumberColumn("Unit Cost", format="$%.2f"),
        "selling_price": st.column_config.NumberColumn("Selling Price", format="$%.2f"),
        "stock_value": st.column_config.NumberColumn("Stock Value", format="$%.2f"),
        "days_to_expiry": st.column_config.NumberColumn("Days Left"),
        "expiry_status": st.column_config.TextColumn("Expiry Status"),
        "recommended_discount_pct": st.column_config.NumberColumn("Discount", format="%d%%"),
        "discounted_price": st.column_config.NumberColumn("Discounted Price", format="$%.2f"),
    }


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Supermarket waste intelligence</div>
            <h1>Smart Food Waste Reduction AI Agent</h1>
            <p>
                A decision dashboard for inventory health, expiry urgency, demand signals,
                policy-backed discounts, and rescue actions. The interface is optimized for
                scanning, keyboard navigation, readable contrast, and clear data labels.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_faq_chat(rag: RagAssistant) -> None:
    if "faq_messages" not in st.session_state:
        st.session_state.faq_messages = [
            {
                "role": "assistant",
                "content": (
                    "Ask about discount rules, expiry urgency, inventory actions, or why a "
                    "product received a recommendation."
                ),
            }
        ]

    faq_options = [
        "Why was yogurt given a 30% discount?",
        "What discount applies when 7 days are left?",
        "What should I do with overstocked products?",
        "When should products be donated to an NGO?",
    ]
    quick_question = st.selectbox(
        "FAQ quick question",
        faq_options,
        help="Choose a common policy question, then select Ask FAQ.",
    )
    if st.button("Ask FAQ", use_container_width=True):
        answer = rag.answer(quick_question)
        st.session_state.faq_messages.append({"role": "user", "content": quick_question})
        st.session_state.faq_messages.append(
            {
                "role": "assistant",
                "content": answer.answer
                + (f"\n\nSources: {', '.join(answer.sources)}" if answer.sources else ""),
            }
        )

    typed_question = st.chat_input("Type a food waste policy question")
    if typed_question:
        answer = rag.answer(typed_question)
        st.session_state.faq_messages.append({"role": "user", "content": typed_question})
        st.session_state.faq_messages.append(
            {
                "role": "assistant",
                "content": answer.answer
                + (f"\n\nSources: {', '.join(answer.sources)}" if answer.sources else ""),
            }
        )

    for message in st.session_state.faq_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def main() -> None:
    inject_theme()
    render_hero()

    with st.sidebar:
        st.header("Data Controls")
        st.caption("Upload CSV files or keep the bundled demo data.")
        sales_file = st.file_uploader(
            "Upload sales_data.csv",
            type=["csv"],
            help="Required columns include date, product_id, product_name, units_sold, and revenue.",
        )
        inventory_file = st.file_uploader(
            "Upload inventory.csv",
            type=["csv"],
            help=(
                "Required columns include product_id, product_name, category, current_stock, "
                "unit_cost, selling_price, expiry_date, daily_sales_avg, and shelf_life_days."
            ),
        )
        st.success("Demo data is loaded until uploads are provided.")
        st.divider()
        st.subheader("Accessibility")
        st.caption("High-contrast surfaces, visible focus states, descriptive labels, and text status values.")

    sales = uploaded_or_sample(sales_file, DATA_DIR / "sales_data.csv")
    inventory = prepare_inventory(uploaded_or_sample(inventory_file, DATA_DIR / "inventory.csv"))
    expiry = monitor_expiry(inventory)
    rescue = recommend_food_rescue(apply_dynamic_pricing(expiry))
    rag = RagAssistant(KNOWLEDGE_BASE_DIR, VECTOR_DB_DIR)

    all_forecasts = {
        product_id: forecast_product_demand(sales, product_id, periods=7).forecasted_demand
        for product_id in inventory["product_id"].unique()
        if product_id in set(sales["product_id"])
    }
    optimization = optimize_inventory(inventory, all_forecasts)

    dashboard_tab, forecast_tab, waste_tab, ai_tab = st.tabs(
        ["Overview", "Forecasting", "Waste Actions", "FAQ Chatbot"]
    )

    with dashboard_tab:
        render_metric_cards(inventory)
        section_title(
            "Inventory Command Center",
            "Review stock value, expiry pressure, and category-level exposure.",
        )
        st.markdown(
            """
            <div class="insight-strip">
                Prioritize products with short shelf life and high stock coverage. Every table
                includes text status labels so the decision does not depend on color alone.
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns([1.35, 1])
        with left:
            st.dataframe(
                inventory[
                    [
                        "product_name",
                        "category",
                        "current_stock",
                        "unit_cost",
                        "selling_price",
                        "expiry_date",
                        "days_to_expiry",
                        "stock_value",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                column_config=inventory_columns(),
            )

        with right:
            value_by_category = (
                inventory.groupby("category", as_index=False)["stock_value"].sum().sort_values(
                    "stock_value", ascending=False
                )
            )
            fig = px.bar(
                value_by_category,
                x="stock_value",
                y="category",
                orientation="h",
                color="category",
                color_discrete_sequence=["#006b5f", "#3a4a8f", "#b8442d", "#8a5a00", "#52635d"],
                labels={"stock_value": "Inventory Value", "category": "Category"},
                title="Inventory Value by Category",
            )
            st.plotly_chart(style_plot(fig), use_container_width=True)

        section_title("Sales Data Preview", "Latest rows used by the forecasting agent.")
        st.dataframe(sales.head(100), use_container_width=True, hide_index=True)

    with forecast_tab:
        section_title(
            "Demand Forecasting",
            "Compare recent sales with forecasted demand and reorder recommendations.",
        )
        product_options = dict(zip(inventory["product_name"], inventory["product_id"]))
        controls = st.columns([2, 1])
        selected_product = controls[0].selectbox(
            "Product to forecast",
            list(product_options.keys()),
            help="Choose the product used for the demand trend chart.",
        )
        selected_product_id = product_options[selected_product]
        horizon = controls[1].slider(
            "Forecast horizon in days",
            min_value=3,
            max_value=21,
            value=7,
            help="Number of future days to estimate demand for.",
        )

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
        fig = px.line(
            chart_data,
            x="date",
            y="units",
            color="series",
            markers=True,
            color_discrete_map={"Actual Sales": "#006b5f", "Forecast": "#b8442d"},
            labels={"units": "Units", "date": "Date"},
            title=f"Sales Trend and Forecast for {forecast.product_name}",
        )
        st.plotly_chart(style_plot(fig), use_container_width=True)

        current_stock = int(
            inventory.loc[inventory["product_id"] == selected_product_id, "current_stock"].iloc[0]
        )
        reorder_units = max(0, forecast.forecasted_demand - current_stock)
        cols = st.columns(3)
        cols[0].metric("Current Stock", f"{current_stock:,}")
        cols[1].metric("Forecasted Demand", f"{forecast.forecasted_demand:,}")
        cols[2].metric("Recommended Reorder", f"{reorder_units:,} units")

        section_title("Inventory Optimization", "Overstock, understock, and replenishment guidance.")
        st.dataframe(optimization, use_container_width=True, hide_index=True)

    with waste_tab:
        section_title(
            "Waste Reduction Actions",
            "Expiry monitoring, dynamic pricing, and rescue recommendations in one workflow.",
        )
        critical_count = int((rescue["expiry_status"] == "Critical").sum())
        warning_count = int((rescue["expiry_status"] == "Warning").sum())
        attention_count = int((rescue["expiry_status"] == "Attention").sum())
        st.markdown(
            f"""
            <div class="status-row">
                <div class="status-box critical"><strong>{critical_count}</strong><span>Critical products: expiry within 3 days</span></div>
                <div class="status-box warning"><strong>{warning_count}</strong><span>Warning products: expiry within 7 days</span></div>
                <div class="status-box attention"><strong>{attention_count}</strong><span>Attention products: expiry within 15 days</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        section_title("Expiry Monitoring", "Products sorted by urgency and days remaining.")
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
            column_config=inventory_columns(),
        )

        pricing_col, rescue_col = st.columns(2)
        with pricing_col:
            section_title("Dynamic Pricing", "Policy-backed markdowns by expiry window.")
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
                column_config=inventory_columns(),
            )
        with rescue_col:
            section_title("Food Rescue", "Operational action for each risky product.")
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
                column_config=inventory_columns(),
            )

    with ai_tab:
        section_title(
            "FAQ Chatbot",
            "Ask policy questions and receive grounded answers with cited source files.",
        )
        chat_col, report_col = st.columns([0.9, 1.1])
        with chat_col:
            render_faq_chat(rag)
        with report_col:
            st.markdown("#### Supervisor Recommendation Report")
            report = final_recommendation_report(sales, inventory, KNOWLEDGE_BASE_DIR, VECTOR_DB_DIR)
            st.dataframe(report, use_container_width=True, hide_index=True)
            csv = report.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Final Recommendation Report",
                data=csv,
                file_name="final_recommendation_report.csv",
                mime="text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
