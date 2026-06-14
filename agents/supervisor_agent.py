from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict, Dict, Any, List

import pandas as pd

from agents.expiry_agent import monitor_expiry
from agents.forecasting_agent import forecast_product_demand
from agents.inventory_agent import optimize_inventory
from agents.pricing_agent import apply_dynamic_pricing
from agents.rag_agent import RagAssistant
from agents.rescue_agent import recommend_food_rescue

from langgraph.graph import StateGraph, END
from crewai.llms.base_llm import BaseLLM
from crewai import Agent, Task, Crew

# Global variable to store orchestration logs from the last run
LAST_RUN_LOGS: List[str] = []


def get_last_execution_logs() -> List[str]:
    """Retrieve the step-by-step agent logs from the last LangGraph/CrewAI execution."""
    return LAST_RUN_LOGS


# -------------------------------------------------------------
# 1. Custom Local Policy LLM for CrewAI
# -------------------------------------------------------------
class LocalPolicyLLM(BaseLLM):
    rag: Any = None

    def call(self, messages: List[Dict[str, str]], callbacks=None, **kwargs) -> str:
        # Extract user prompt from the messages list
        user_msg = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
        
        # Ground reasoning in the offline policy knowledge base
        if self.rag and user_msg:
            return self.rag.answer(user_msg).answer
        return "Action recommended in accordance with standard store markdown guidelines."


# -------------------------------------------------------------
# 2. State definition for LangGraph
# -------------------------------------------------------------
class SupermarketState(TypedDict):
    sales: pd.DataFrame
    inventory: pd.DataFrame
    forecasts: Dict[str, int]
    optimization_report: pd.DataFrame
    expiry_report: pd.DataFrame
    pricing_report: pd.DataFrame
    rescue_report: pd.DataFrame
    policy_basis_dict: Dict[str, str]  # Maps product_id -> justification text
    final_report: pd.DataFrame
    knowledge_base_dir: Path
    vector_db_dir: Path
    logs: List[str]


# -------------------------------------------------------------
# 3. LangGraph Nodes representing Agentic State Transitions
# -------------------------------------------------------------
def expiry_node(state: SupermarketState) -> Dict[str, Any]:
    state["logs"].append("[LangGraph: Expiry Agent] Analyzing inventory expiry dates...")
    monitored = monitor_expiry(state["inventory"])
    state["logs"].append(f"[LangGraph: Expiry Agent] Completed classifying {len(monitored)} products.")
    return {"expiry_report": monitored, "logs": state["logs"]}


def forecasting_node(state: SupermarketState) -> Dict[str, Any]:
    state["logs"].append("[LangGraph: Forecasting Agent] Estimating 7-day future demand from sales historical trend...")
    sales = state["sales"]
    inventory = state["inventory"]
    
    forecasts = {}
    for product_id in inventory["product_id"].unique():
        if product_id in set(sales["product_id"]):
            forecast = forecast_product_demand(sales, product_id, periods=7)
            forecasts[product_id] = forecast.forecasted_demand
            
    state["logs"].append(f"[LangGraph: Forecasting Agent] Generated demand forecasts for {len(forecasts)} products.")
    return {"forecasts": forecasts, "logs": state["logs"]}


def inventory_node(state: SupermarketState) -> Dict[str, Any]:
    state["logs"].append("[LangGraph: Inventory Agent] Evaluating stock levels against forecasted demand + safety margin...")
    optimization = optimize_inventory(state["inventory"], state["forecasts"])
    state["logs"].append("[LangGraph: Inventory Agent] Compiled stock recommendations.")
    return {"optimization_report": optimization, "logs": state["logs"]}


def pricing_node(state: SupermarketState) -> Dict[str, Any]:
    state["logs"].append("[LangGraph: Pricing Agent] Computing dynamic discounts based on expiry windows...")
    pricing = apply_dynamic_pricing(state["expiry_report"])
    state["logs"].append("[LangGraph: Pricing Agent] Updated pricing calculations.")
    return {"pricing_report": pricing, "logs": state["logs"]}


def rescue_node(state: SupermarketState) -> Dict[str, Any]:
    state["logs"].append("[LangGraph: Food Rescue Agent] Formulating store actions and NGO donation proposals...")
    rescue = recommend_food_rescue(state["pricing_report"])
    state["logs"].append("[LangGraph: Food Rescue Agent] Food rescue suggestions completed.")
    return {"rescue_report": rescue, "logs": state["logs"]}


def crewai_policy_node(state: SupermarketState) -> Dict[str, Any]:
    state["logs"].append("[LangGraph] Transitioning control to CrewAI for policy compliance verification...")
    
    rag = RagAssistant(state["knowledge_base_dir"], state["vector_db_dir"])
    local_llm = LocalPolicyLLM(model="local-supermarket-sop", rag=rag)
    
    # Initialize a CrewAI Role-Playing Agent
    policy_auditor = Agent(
        role="Supermarket Policy Auditor",
        goal="Audit markdown proposals and ensure they comply with safety and discount SOPs.",
        backstory="An expert auditor who verifies that all discount percentages correspond correctly to expiry horizons.",
        llm=local_llm,
        verbose=False
    )
    
    policy_basis_dict = {}
    rescue_df = state["rescue_report"]
    
    state["logs"].append("[CrewAI] Created Agent 'Supermarket Policy Auditor' using LocalPolicyLLM.")
    
    for idx, row in rescue_df.iterrows():
        p_name = row["product_name"]
        discount_pct = row["recommended_discount_pct"]
        p_id = row["product_id"]
        
        # Instantiate CrewAI Task for auditing policy rules
        task_desc = f"Determine the policy basis for {p_name} receiving a {discount_pct}% discount."
        task = Task(
            description=task_desc,
            expected_output="A single sentence policy explanation grounded in store SOPs.",
            agent=policy_auditor
        )
        
        crew = Crew(
            agents=[policy_auditor],
            tasks=[task],
            verbose=False
        )
        
        state["logs"].append(f"[CrewAI Agent: Policy Auditor] Auditing rule compliance for product: {p_name}...")
        result = crew.kickoff()
        policy_basis_dict[p_id] = str(result).strip()
        
    state["logs"].append("[LangGraph] Nodes finished. Handing control back to state graph compiler...")
    return {"policy_basis_dict": policy_basis_dict, "logs": state["logs"]}


def compile_report_node(state: SupermarketState) -> Dict[str, Any]:
    state["logs"].append("[LangGraph: Report Compiler] Merging agent outputs into final dashboard view...")
    
    rescue_df = state["rescue_report"]
    opt_df = state["optimization_report"]
    policy_dict = state["policy_basis_dict"]
    
    report = rescue_df.merge(
        opt_df[
            [
                "product_id",
                "forecasted_demand",
                "recommended_reorder",
                "inventory_status",
                "recommendation",
            ]
        ],
        on="product_id",
        how="left",
    )
    
    report["policy_basis"] = report["product_id"].map(policy_dict)
    
    final_df = report[
        [
            "product_name",
            "current_stock",
            "forecasted_demand",
            "recommended_reorder",
            "inventory_status",
            "expiry_status",
            "recommended_discount_pct",
            "food_rescue_action",
            "policy_basis",
        ]
    ]
    
    state["logs"].append("[LangGraph: Report Compiler] Final recommendation report prepared.")
    return {"final_report": final_df, "logs": state["logs"]}


# -------------------------------------------------------------
# 4. Main Entry Point: Stateful Orchestration Loop
# -------------------------------------------------------------
def final_recommendation_report(
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
    knowledge_base_dir: Path,
    vector_db_dir: Path,
) -> pd.DataFrame:
    """Computes the final recommendation report using a LangGraph state machine and CrewAI agents."""
    
    # Construct the graph flow
    workflow = StateGraph(SupermarketState)
    
    workflow.add_node("expiry", expiry_node)
    workflow.add_node("forecasting", forecasting_node)
    workflow.add_node("inventory_opt", inventory_node)
    workflow.add_node("pricing", pricing_node)
    workflow.add_node("rescue", rescue_node)
    workflow.add_node("crewai_policy", crewai_policy_node)
    workflow.add_node("compile_report", compile_report_node)
    
    workflow.set_entry_point("expiry")
    workflow.add_edge("expiry", "forecasting")
    workflow.add_edge("forecasting", "inventory_opt")
    workflow.add_edge("inventory_opt", "pricing")
    workflow.add_edge("pricing", "rescue")
    workflow.add_edge("rescue", "crewai_policy")
    workflow.add_edge("crewai_policy", "compile_report")
    workflow.add_edge("compile_report", END)
    
    # Compile graph
    app = workflow.compile()
    
    initial_state = {
        "sales": sales,
        "inventory": inventory,
        "forecasts": {},
        "optimization_report": pd.DataFrame(),
        "expiry_report": pd.DataFrame(),
        "pricing_report": pd.DataFrame(),
        "rescue_report": pd.DataFrame(),
        "policy_basis_dict": {},
        "final_report": pd.DataFrame(),
        "knowledge_base_dir": knowledge_base_dir,
        "vector_db_dir": vector_db_dir,
        "logs": []
    }
    
    # Run the orchestration graph
    final_output = app.invoke(initial_state)
    
    # Save logs to a global store so the UI can retrieve and display them
    global LAST_RUN_LOGS
    LAST_RUN_LOGS = final_output.get("logs", [])
    
    return final_output["final_report"]
