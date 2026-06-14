# Smart Food Waste Reduction AI Agent for Supermarkets

Hackathon-ready Streamlit application for supermarket inventory visibility, demand planning, waste reduction, and policy-aware recommendations.

## GitHub Milestones

1. **Foundation & Dashboard (Day 1)** - Streamlit app, folder structure, sample data uploads, dashboard home page, and KPI cards.
2. **Machine Learning Engine (Day 2)** - Demand forecasting, sales trend analysis, inventory optimization, and reorder recommendations.
3. **AI Waste Reduction System (Day 3)** - Expiry monitoring, dynamic pricing, and food rescue recommendations.
4. **RAG + Agentic AI + Deployment (Day 4)** - ChromaDB knowledge base, LangChain-style assistant, supervisor workflow, documentation, and deployment notes.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Phase 1 Status

- Upload `sales_data.csv` and `inventory.csv`
- Dashboard home page
- KPI cards for total products, inventory value, expiring products, and waste risk score
- Bundled sample datasets in `data/`

## Phase 2 Status

- Demand forecasting agent using a lightweight trend model suitable for fast demos
- Sales trend visualization
- Inventory optimization agent
- Overstock, understock, and reorder recommendations

## Phase 3 Status

- Expiry monitoring agent with Critical, Warning, Attention, and Normal bands
- Dynamic pricing agent using 30%, 20%, and 10% discount rules
- Food rescue recommendation agent for discount campaigns, bundles, NGO donation, and store transfer

## Phase 4 Status

- RAG assistant over policy documents in `knowledge_base/`
- ChromaDB persistent vector database in `vector_db/` when dependencies are installed
- Lightweight RAG flow with ChromaDB persistence when available
- Supervisor agent that combines forecasting, inventory, expiry, pricing, rescue, and RAG outputs
- Final recommendation report download from Streamlit

## Architecture

```text
Supervisor Agent
       |
       |---- Forecasting Agent
       |
       |---- Inventory Agent
       |
       |---- Expiry Agent
       |
       |---- Pricing Agent
       |
       |---- RAG Assistant Agent
       |
       V
 Final Recommendation Report
```

## Deployment

1. Push this repository to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Select `app.py` as the entry point.
4. Add demo screenshots and a walkthrough video under `assets/screenshots/` before final submission.
