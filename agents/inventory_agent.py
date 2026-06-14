from __future__ import annotations

import pandas as pd


def optimize_inventory(
    inventory: pd.DataFrame, forecasts: dict[str, int], safety_stock_days: int = 2
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for item in inventory.to_dict("records"):
        product_id = str(item["product_id"])
        daily_sales = max(float(item.get("daily_sales_avg", 0)), 1)
        forecasted_demand = int(forecasts.get(product_id, round(daily_sales * 7)))
        safety_stock = int(round(daily_sales * safety_stock_days))
        recommended_stock = forecasted_demand + safety_stock
        current_stock = int(item["current_stock"])
        gap = recommended_stock - current_stock

        if current_stock > recommended_stock * 1.25:
            status = "Overstock"
            recommendation = "Run discount campaign or transfer surplus"
            reorder_units = 0
        elif gap > 0:
            status = "Understock"
            recommendation = f"Reorder {gap} units"
            reorder_units = gap
        else:
            status = "Healthy"
            recommendation = "Maintain current replenishment plan"
            reorder_units = 0

        rows.append(
            {
                "product_id": product_id,
                "product_name": item["product_name"],
                "current_stock": current_stock,
                "forecasted_demand": forecasted_demand,
                "recommended_reorder": reorder_units,
                "inventory_status": status,
                "recommendation": recommendation,
            }
        )

    return pd.DataFrame(rows)
