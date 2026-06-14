from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


@dataclass
class ForecastResult:
    product_id: str
    product_name: str
    forecast: pd.DataFrame
    forecasted_demand: int


def _model() -> object:
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.08,
            objective="reg:squarederror",
            random_state=42,
        )
    except Exception:
        return GradientBoostingRegressor(random_state=42)


def forecast_product_demand(
    sales: pd.DataFrame, product_id: str, periods: int = 7
) -> ForecastResult:
    product_sales = sales[sales["product_id"] == product_id].copy()
    if product_sales.empty:
        raise ValueError(f"No sales data found for product_id={product_id}")

    product_sales["date"] = pd.to_datetime(product_sales["date"])
    product_sales = product_sales.sort_values("date")
    product_sales["day_index"] = np.arange(len(product_sales))
    product_sales["day_of_week"] = product_sales["date"].dt.dayofweek

    features = product_sales[["day_index", "day_of_week"]]
    target = product_sales["units_sold"]

    estimator = _model() if len(product_sales) >= 8 else LinearRegression()
    estimator.fit(features, target)

    last_date = product_sales["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods)
    future = pd.DataFrame(
        {
            "date": future_dates,
            "day_index": np.arange(len(product_sales), len(product_sales) + periods),
            "day_of_week": future_dates.dayofweek,
        }
    )
    future["forecast_units"] = np.maximum(
        0, np.round(estimator.predict(future[["day_index", "day_of_week"]])).astype(int)
    )

    return ForecastResult(
        product_id=product_id,
        product_name=str(product_sales["product_name"].iloc[0]),
        forecast=future[["date", "forecast_units"]],
        forecasted_demand=int(future["forecast_units"].sum()),
    )


def sales_trend(sales: pd.DataFrame, product_id: str) -> pd.DataFrame:
    product_sales = sales[sales["product_id"] == product_id].copy()
    product_sales["date"] = pd.to_datetime(product_sales["date"])
    return product_sales.sort_values("date")[["date", "units_sold"]]
