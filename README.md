# Smart Food Waste Reduction AI Agent for Supermarkets

## Project Overview

Smart Food Waste Reduction AI Agent is an AI-powered inventory intelligence platform designed to help supermarkets reduce food waste, optimize stock levels, and improve profitability through data-driven decision-making.

The system leverages historical sales data and inventory information to identify waste risks, forecast product demand, recommend inventory actions, and generate actionable insights for supermarket managers.

---

## Problem Statement

Food retailers frequently face challenges such as:

* Overstocking perishable products
* Product expiration before sale
* Inventory holding costs
* Revenue loss due to inaccurate demand planning

This project addresses these challenges by combining Machine Learning, Agentic AI, and Business Analytics to improve inventory efficiency and reduce food waste.

---

## Dataset Description

### Sales Dataset

File: `final_large_sales_data.csv`

Columns:

* date
* product_id
* product_name
* units_sold
* revenue

Purpose:

* Sales trend analysis
* Demand forecasting
* Product performance evaluation

### Inventory Dataset

File: `final_large_inventory.csv`

Columns:

* product_id
* product_name
* category
* current_stock
* unit_cost
* selling_price
* expiry_date
* daily_sales_avg
* shelf_life_days
* supplier

Purpose:

* Inventory monitoring
* Waste risk assessment
* Expiry management
* Stock optimization

---

## Key Features

### Demand Forecasting Agent

Uses historical sales records to:

* Analyze sales patterns
* Forecast future demand
* Identify high-demand products
* Support inventory planning

### Inventory Optimization Agent

Analyzes:

* Current stock levels
* Daily sales averages
* Product demand

Generates:

* Reorder recommendations
* Overstock alerts
* Understock alerts

### Expiry Monitoring Agent

Monitors:

* Expiry dates
* Shelf life
* Current stock

Risk Categories:

* Critical Risk
* Medium Risk
* Low Risk

Provides proactive alerts for products nearing expiry.

### Waste Reduction Agent

Identifies products likely to become waste and recommends:

* Promotional campaigns
* Inventory transfers
* Stock reduction strategies
* Clearance sales

### Profitability Analysis

Calculates:

* Revenue generated
* Inventory value
* Potential losses due to expiry
* Estimated savings from recommendations

### AI Insights Dashboard

Displays:

* Total Products
* Total Inventory Value
* Total Revenue
* Products Near Expiry
* Waste Risk Score
* Recommended Actions

---

## Technology Stack

### Frontend

* Streamlit
* Plotly

### Backend

* Python
* Pandas
* NumPy

### Machine Learning

* Prophet
* XGBoost
* Scikit-Learn

### Agentic AI

* LangGraph
* CrewAI

### RAG

* LangChain
* ChromaDB

### Database

* SQLite

### LLM

* OpenAI GPT
* Ollama (Llama 3)

---

## Project Workflow

Sales Data
↓
Demand Forecasting Agent
↓
Inventory Analysis Agent
↓
Expiry Monitoring Agent
↓
Waste Reduction Agent
↓
Business Insights Engine
↓
Streamlit Dashboard

---

## Business Benefits

* Reduce food waste
* Improve inventory turnover
* Increase profitability
* Optimize purchasing decisions
* Improve sustainability metrics
* Enable data-driven inventory management

---

## Future Enhancements

* Dynamic pricing recommendations
* Supplier optimization
* Multi-store inventory balancing
* Real-time stock monitoring
* Food donation recommendation engine
* RAG-powered supermarket assistant

---

## Installation

```bash
[git clone https://github.com/your-username/smart-food-waste-ai-agent.git](https://github.com/sivamohanbabu/AI-Agent-for-Supermarkets.git)

cd smart-food-waste-ai-agent

pip install -r requirements.txt

streamlit run app.py
```

## Author

Siva Veludurthi

AI | Data Analytics | Education | Workforce Development

Building AI-powered solutions for sustainable retail and inventory optimization.
