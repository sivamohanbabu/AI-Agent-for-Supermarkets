from __future__ import annotations

import pandas as pd


def classify_expiry(days_to_expiry: int) -> str:
    if days_to_expiry <= 3:
        return "Critical"
    if days_to_expiry <= 7:
        return "Warning"
    if days_to_expiry <= 15:
        return "Attention"
    return "Normal"


def monitor_expiry(inventory: pd.DataFrame) -> pd.DataFrame:
    monitored = inventory.copy()
    monitored["expiry_status"] = monitored["days_to_expiry"].apply(classify_expiry)
    monitored["priority_rank"] = monitored["expiry_status"].map(
        {"Critical": 1, "Warning": 2, "Attention": 3, "Normal": 4}
    )
    return monitored.sort_values(["priority_rank", "days_to_expiry", "current_stock"])
