from __future__ import annotations

import pandas as pd


def rescue_recommendation(row: pd.Series) -> str:
    days = int(row["days_to_expiry"])
    stock = int(row["current_stock"])
    daily_sales = max(float(row.get("daily_sales_avg", 1)), 1)
    cover_days = stock / daily_sales

    if days < 0:
        return "Dispose/Discard (Expired)"
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


def calculate_sustainability_impact(rescue_df: pd.DataFrame) -> dict[str, float]:
    co2_saved = 0.0
    water_saved = 0.0
    
    for idx, row in rescue_df.iterrows():
        action = row.get("food_rescue_action", "Monitor")
        if action in {"NGO Donation", "30% Discount Campaign", "10% Discount Campaign", "Bundle Offers", "Transfer to Nearby Store"}:
            category = str(row.get("category", "")).lower()
            stock = float(row.get("current_stock", 0))
            
            if "meat" in category:
                co2_factor = 27.0
                water_factor = 15400.0
            elif "dairy" in category:
                co2_factor = 3.2
                water_factor = 1000.0
            elif "produce" in category:
                co2_factor = 2.0
                water_factor = 250.0
            elif "bakery" in category:
                co2_factor = 1.0
                water_factor = 100.0
            else:
                co2_factor = 1.5
                water_factor = 150.0
                
            co2_saved += stock * co2_factor
            water_saved += stock * water_factor
            
    return {"co2_saved_kg": round(co2_saved, 1), "water_saved_liters": round(water_saved, 1)}
