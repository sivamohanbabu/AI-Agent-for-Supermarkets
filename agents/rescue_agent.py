from __future__ import annotations

import pandas as pd


def rescue_recommendation(row: pd.Series) -> str:
    days = int(row["days_to_expiry"])
    stock = int(row["current_stock"])
    daily_sales = max(float(row.get("daily_sales_avg", 1)), 1)
    cover_days = stock / daily_sales

    if days <= 3 and cover_days > 2:
        return "NGO Donation"
    if days <= 3:
        return "30% Discount Campaign"
    if days <= 7 and cover_days > 4:
        return "Bundle Offers"
    if days <= 15 and cover_days > 7:
        return "Transfer to Nearby Store"
    if days <= 15:
        return "10% Discount Campaign"
    return "Monitor"


def recommend_food_rescue(inventory: pd.DataFrame) -> pd.DataFrame:
    recommendations = inventory.copy()
    recommendations["food_rescue_action"] = recommendations.apply(rescue_recommendation, axis=1)
    return recommendations
