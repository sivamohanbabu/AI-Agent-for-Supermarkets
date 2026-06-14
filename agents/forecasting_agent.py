from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ForecastResult:
    product_id: str
    product_name: str
    forecast: pd.DataFrame
    forecasted_demand: int


def _trend_forecast(values: list[float], periods: int) -> list[int]:
    if not values:
        return [0] * periods
    recent = values[-7:]
    moving_average = sum(recent) / len(recent)
    if len(values) >= 2:
        trend = (values[-1] - values[0]) / max(len(values) - 1, 1)
    else:
        trend = 0
    return [max(0, round(moving_average + trend * step)) for step in range(1, periods + 1)]


def forecast_product_demand(
    sales: pd.DataFrame, product_id: str, periods: int = 7
) -> ForecastResult:
    product_sales = sales[sales["product_id"] == product_id].copy()
    if product_sales.empty:
        raise ValueError(f"No sales data found for product_id={product_id}")

    product_sales["date"] = pd.to_datetime(product_sales["date"])
    product_sales = product_sales.sort_values("date")
    last_date = product_sales["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods)
    future = pd.DataFrame(
        {
            "date": future_dates,
            "forecast_units": _trend_forecast(
                product_sales["units_sold"].astype(float).tolist(), periods
            ),
        }
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
