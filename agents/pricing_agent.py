from __future__ import annotations

import pandas as pd


def recommended_discount(days_to_expiry: int) -> int:
    if days_to_expiry < 0:
        return 0
    if days_to_expiry <= 3:
        return 30
    if days_to_expiry <= 7:
        return 20
    if days_to_expiry <= 15:
        return 10
    return 0


def apply_dynamic_pricing(inventory: pd.DataFrame) -> pd.DataFrame:
    priced = inventory.copy()
    priced["recommended_discount_pct"] = priced["days_to_expiry"].apply(recommended_discount)
    priced["discounted_price"] = (
        priced["selling_price"] * (1 - priced["recommended_discount_pct"] / 100)
    ).round(2)
    # If expired, set discounted price to 0.00 indicating removal from sale
    priced.loc[priced["days_to_expiry"] < 0, "discounted_price"] = 0.00
    return priced

