from __future__ import annotations

from pathlib import Path

import pandas as pd

from agents.expiry_agent import monitor_expiry
from agents.forecasting_agent import forecast_product_demand
from agents.inventory_agent import optimize_inventory
from agents.pricing_agent import apply_dynamic_pricing
from agents.rag_agent import RagAssistant
from agents.rescue_agent import recommend_food_rescue


def final_recommendation_report(
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
    knowledge_base_dir: Path,
    vector_db_dir: Path,
) -> pd.DataFrame:
    forecasts = {
        product_id: forecast_product_demand(sales, product_id, periods=7).forecasted_demand
        for product_id in inventory["product_id"].unique()
        if product_id in set(sales["product_id"])
    }
    optimization = optimize_inventory(inventory, forecasts)
    waste = recommend_food_rescue(apply_dynamic_pricing(monitor_expiry(inventory)))
    rag = RagAssistant(knowledge_base_dir, vector_db_dir)

    report = waste.merge(
        optimization[
            [
                "product_id",
                "forecasted_demand",
                "recommended_reorder",
                "inventory_status",
                "recommendation",
            ]
        ],
        on="product_id",
        how="left",
    )
    report["policy_basis"] = report.apply(
        lambda row: rag.answer(
            f"Why was {row['product_name']} given a {row['recommended_discount_pct']}% discount?"
        ).answer,
        axis=1,
    )
    return report[
        [
            "product_name",
            "current_stock",
            "forecasted_demand",
            "recommended_reorder",
            "inventory_status",
            "expiry_status",
            "recommended_discount_pct",
            "food_rescue_action",
            "policy_basis",
        ]
    ]
