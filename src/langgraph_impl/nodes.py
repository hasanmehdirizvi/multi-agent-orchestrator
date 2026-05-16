"""LangGraph node definitions for the customer service graph.

Each node is a pure function: State -> State. This makes the system
fully testable and observable - you can snapshot state at any point
and replay from there.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

from src.common.config import get_config
from src.langgraph_impl.state import AgentState

_config = get_config()


def _get_llm(temperature: float = 0.1) -> ChatBedrock:
    """Create a Bedrock LLM instance."""
    return ChatBedrock(
        model_id=_config.specialist_model.model_id,
        region_name=_config.specialist_model.region,
        model_kwargs={"temperature": temperature, "max_tokens": 4096},
    )


# ---------------------------------------------------------------------------
# Node: Router
# ---------------------------------------------------------------------------

def router_node(state: AgentState) -> dict[str, Any]:
    """Classify intent and determine routing.

    This node runs the LLM with a classification prompt and writes
    the intent, confidence, and reasoning back to state.
    """
    llm = _get_llm(temperature=0.0)

    messages = [
        SystemMessage(content="""Classify this customer query into one intent.
Respond with valid JSON only:
{"intent": "billing|technical|escalation|general", "confidence": 0.0-1.0, "reasoning": "..."}"""),
        HumanMessage(content=f"Customer {state['customer_id']}: {state['query_text']}"),
    ]

    response = llm.invoke(messages)

    try:
        classification = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        classification = {
            "intent": "general",
            "confidence": 0.5,
            "reasoning": "Failed to parse classification",
        }

    return {
        "intent": classification["intent"],
        "confidence": classification["confidence"],
        "routing_reasoning": classification["reasoning"],
        "current_node": "router",
        "messages": [HumanMessage(content=state["query_text"])],
    }


# ---------------------------------------------------------------------------
# Node: Billing Specialist
# ---------------------------------------------------------------------------

def billing_node(state: AgentState) -> dict[str, Any]:
    """Handle billing inquiries with CRM context."""
    llm = _get_llm()

    system_prompt = """You are a billing specialist. The customer has a billing question.
Use the customer context provided to give a specific, helpful response.
If the issue cannot be resolved automatically, indicate escalation is needed."""

    context = f"""Customer ID: {state['customer_id']}
Customer Record: {json.dumps(state.get('customer_record') or {}, default=str)}
Billing History: {json.dumps(state.get('billing_history') or [], default=str)}
Query: {state['query_text']}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)

    # Check for escalation triggers
    query_lower = state["query_text"].lower()
    needs_escalation = any(
        w in query_lower for w in ["legal", "lawyer", "cancel everything", "regulator"]
    )

    return {
        "response_text": response.content,
        "actions_taken": ["crm_lookup", "billing_analysis"],
        "requires_escalation": needs_escalation,
        "current_node": "billing",
    }


# ---------------------------------------------------------------------------
# Node: Technical Specialist
# ---------------------------------------------------------------------------

def technical_node(state: AgentState) -> dict[str, Any]:
    """Handle technical support with diagnostic workflow."""
    llm = _get_llm()

    system_prompt = """You are a technical support specialist. Diagnose the issue and provide:
1. Issue Summary
2. Likely Root Cause
3. Immediate Mitigation Steps
4. Resolution Timeline

Be specific and actionable."""

    context = f"""Customer ID: {state['customer_id']}
Customer Record: {json.dumps(state.get('customer_record') or {}, default=str)}
Technical Query: {state['query_text']}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)

    query_lower = state["query_text"].lower()
    needs_escalation = any(
        w in query_lower for w in ["outage", "data loss", "breach", "production down"]
    )

    return {
        "response_text": response.content,
        "actions_taken": ["diagnostic_analysis", "ticket_created"],
        "requires_escalation": needs_escalation,
        "current_node": "technical",
    }


# ---------------------------------------------------------------------------
# Node: Escalation
# ---------------------------------------------------------------------------

def escalation_node(state: AgentState) -> dict[str, Any]:
    """Prepare escalation package for human handoff."""
    llm = _get_llm(temperature=0.0)

    system_prompt = """You are preparing an escalation package for a human agent.
Summarize the situation, what has been tried, and what the human agent needs to do.
Determine priority (critical/high/medium/low) and the correct team."""

    context = f"""Customer: {state['customer_id']}
Original Query: {state['query_text']}
Intent: {state.get('intent', 'unknown')}
Previous Response: {state.get('response_text', 'None')}
Actions Taken: {state.get('actions_taken', [])}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)

    # Determine priority and team
    query_lower = state["query_text"].lower()
    if any(w in query_lower for w in ["breach", "security", "legal"]):
        priority, team = "critical", "security"
    elif any(w in query_lower for w in ["outage", "production", "sla"]):
        priority, team = "high", "engineering_l2"
    elif any(w in query_lower for w in ["cancel", "churn"]):
        priority, team = "medium", "retention"
    else:
        priority, team = "medium", "general_support"

    return {
        "escalation_priority": priority,
        "escalation_team": team,
        "escalation_reason": response.content,
        "current_node": "escalation",
    }


# ---------------------------------------------------------------------------
# Node: CRM Enrichment (pre-processing node)
# ---------------------------------------------------------------------------

def crm_enrichment_node(state: AgentState) -> dict[str, Any]:
    """Enrich state with CRM data before specialist processing.

    This demonstrates LangGraph's ability to have pre-processing nodes
    that gather context before the main logic runs.
    """
    from src.strands_impl.tools.crm_lookup import _CRM_DB

    customer_id = state["customer_id"]
    record = _CRM_DB.get(customer_id)

    billing_history = None
    if record:
        monthly_spend = record["monthly_spend"]
        billing_history = [
            {"month": f"2026-{5 - i:02d}", "amount": monthly_spend, "status": "paid"}
            for i in range(3)
        ]

    return {
        "customer_record": record,
        "billing_history": billing_history,
    }
