import unittest
import pandas as pd
from datetime import date, timedelta
from agents.expiry_agent import classify_expiry, monitor_expiry
from agents.pricing_agent import recommended_discount, apply_dynamic_pricing
from agents.rescue_agent import rescue_recommendation, recommend_food_rescue
from agents.inventory_agent import optimize_inventory

class TestSupermarketAgents(unittest.TestCase):

    def setUp(self):
        # Create a mock inventory DataFrame
        self.inventory_df = pd.DataFrame([
            {
                "product_id": "P001",
                "product_name": "Fresh Milk",
                "category": "Dairy",
                "current_stock": 100,
                "unit_cost": 1.0,
                "selling_price": 2.0,
                "days_to_expiry": 2, # Critical
                "daily_sales_avg": 20.0,
                "shelf_life_days": 7
            },
            {
                "product_id": "P002",
                "product_name": "Old Bread",
                "category": "Bakery",
                "current_stock": 50,
                "unit_cost": 0.5,
                "selling_price": 1.5,
                "days_to_expiry": -2, # Expired!
                "daily_sales_avg": 10.0,
                "shelf_life_days": 5
            },
            {
                "product_id": "P003",
                "product_name": "Fresh Apple",
                "category": "Produce",
                "current_stock": 200,
                "unit_cost": 0.2,
                "selling_price": 0.6,
                "days_to_expiry": 10, # Attention
                "daily_sales_avg": 15.0,
                "shelf_life_days": 15
            }
        ])

    def test_classify_expiry(self):
        self.assertEqual(classify_expiry(-5), "Expired")
        self.assertEqual(classify_expiry(0), "Critical")
        self.assertEqual(classify_expiry(3), "Critical")
        self.assertEqual(classify_expiry(5), "Warning")
        self.assertEqual(classify_expiry(7), "Warning")
        self.assertEqual(classify_expiry(12), "Attention")
        self.assertEqual(classify_expiry(20), "Normal")

    def test_monitor_expiry(self):
        monitored = monitor_expiry(self.inventory_df)
        self.assertEqual(monitored.iloc[0]["expiry_status"], "Expired")
        self.assertEqual(monitored.iloc[1]["expiry_status"], "Critical")
        self.assertEqual(monitored.iloc[2]["expiry_status"], "Attention")

    def test_recommended_discount(self):
        self.assertEqual(recommended_discount(-1), 0)
        self.assertEqual(recommended_discount(2), 30)
        self.assertEqual(recommended_discount(5), 20)
        self.assertEqual(recommended_discount(12), 10)
        self.assertEqual(recommended_discount(20), 0)

    def test_apply_dynamic_pricing(self):
        priced = apply_dynamic_pricing(self.inventory_df)
        # Old Bread (P002) is expired (days_to_expiry = -2) -> discounted_price should be 0.00
        bread_price = priced.loc[priced["product_id"] == "P002", "discounted_price"].values[0]
        self.assertEqual(bread_price, 0.00)
        
        # Fresh Milk (P001) is critical (days_to_expiry = 2) -> 30% discount on 2.0 -> 1.40
        milk_price = priced.loc[priced["product_id"] == "P001", "discounted_price"].values[0]
        self.assertEqual(milk_price, 1.40)

    def test_rescue_recommendation(self):
        # Expired item -> Dispose
        expired_row = pd.Series({"days_to_expiry": -1, "current_stock": 10, "daily_sales_avg": 5})
        self.assertEqual(rescue_recommendation(expired_row), "Dispose/Discard (Expired)")
        
        # Critical and overstocked -> NGO Donation
        donation_row = pd.Series({"days_to_expiry": 2, "current_stock": 50, "daily_sales_avg": 5})
        self.assertEqual(rescue_recommendation(donation_row), "NGO Donation")

    def test_optimize_inventory(self):
        forecasts = {"P001": 150, "P002": 30, "P003": 100}
        opt = optimize_inventory(self.inventory_df, forecasts)
        
        # P001: stock 100, forecast 150, reorder should be > 0 (Understock)
        p001_status = opt.loc[opt["product_id"] == "P001", "inventory_status"].values[0]
        self.assertEqual(p001_status, "Understock")
        
        # P003: stock 200, forecast 100 -> reorder 0 (Overstock)
        p003_status = opt.loc[opt["product_id"] == "P003", "inventory_status"].values[0]
        self.assertEqual(p003_status, "Overstock")

if __name__ == "__main__":
    unittest.main()
