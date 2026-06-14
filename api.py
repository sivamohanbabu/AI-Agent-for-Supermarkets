from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents.supervisor_agent import final_recommendation_report
from agents.rag_agent import RagAssistant
from app import prepare_inventory

# Initialize FastAPI
app = FastAPI(
    title="Smart Food Waste Reduction AI Agent API",
    description="Microservice exposing the LangGraph orchestration flow and CrewAI policy grounding rules.",
    version="1.0.0"
)

# Directories
BASE_DIR = Path(__file__).parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# Model Schemas
class ChatQuery(BaseModel):
    query: str

class SalesRecord(BaseModel):
    date: str
    product_id: str
    product_name: str
    units_sold: int
    revenue: float

class InventoryRecord(BaseModel):
    product_id: str
    product_name: str
    category: str
    current_stock: int
    unit_cost: float
    selling_price: float
    expiry_date: str
    daily_sales_avg: float
    shelf_life_days: int
    supplier: str

class RecommendationPayload(BaseModel):
    sales: List[SalesRecord]
    inventory: List[InventoryRecord]


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"status": "online", "service": "Supermarket AI Orchestrator"}


@app.post("/api/chat")
def chat_with_policy(payload: ChatQuery):
    """Query the store policy RAG assistant directly."""
    try:
        rag = RagAssistant(KNOWLEDGE_BASE_DIR, VECTOR_DB_DIR)
        response = rag.answer(payload.query)
        return {
            "query": payload.query,
            "answer": response.answer,
            "sources": response.sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recommendations")
def get_recommendations(payload: RecommendationPayload):
    """Run the stateful LangGraph flow and CrewAI audit crew on provided sales and inventory logs."""
    try:
        # Convert Pydantic lists to DataFrames
        sales_data = [rec.model_dump() for rec in payload.sales]
        inventory_data = [rec.model_dump() for rec in payload.inventory]
        
        if not sales_data or not inventory_data:
            raise HTTPException(status_code=400, detail="Sales and inventory data lists cannot be empty.")
            
        sales_df = pd.DataFrame(sales_data)
        raw_inventory_df = pd.DataFrame(inventory_data)
        
        # Prepare days left and cost calculations
        inventory_df = prepare_inventory(raw_inventory_df)
        
        # Invoke LangGraph flow
        report_df = final_recommendation_report(
            sales_df,
            inventory_df,
            KNOWLEDGE_BASE_DIR,
            VECTOR_DB_DIR
        )
        
        # Return structured recommendations list
        return {
            "total_items_processed": len(report_df),
            "recommendations": report_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
