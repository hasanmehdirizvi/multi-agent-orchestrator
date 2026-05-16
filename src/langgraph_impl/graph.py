"""LangGraph StateGraph implementation with conditional routing.

This module defines the full execution graph for customer service orchestration.
LangGraph excels here because:

1. Explicit control flow - routing logic is visible in the graph definition
2. State checkpointing - pause at any node, resume later (human-in-the-loop)
3. Conditional edges - branching logic is declarative and testable
4. Observability - every state transition is logged with full context
5. Replay - failed runs can be restarted from any checkpoint

Trade-off vs Strands: More boilerplate, but you get deterministic flow control
and built-in persistence. Better for regulated environments where you need
audit trails of every decision.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.common.config import get_config
from src.langgraph_impl.nodes import (
    billing_node,
    crm_enrichment_node,
    escalation_node,
    router_node,
    technical_node,
)
from src.langgraph_impl.state import AgentState

_config = get_config()


def _route_after_classification(state: AgentState) -> str:
    """Conditional edge: route based on classified intent and confidence.

    This function is the decision point in the graph. LangGraph calls it
    after the router node completes and uses the return value to select
    the next edge.
    """
    confidence = state.get("confidence", 0.0)
    intent = state.get("intent", "general")

    # Low confidence -> escalate directly
    if confidence < _config.escalation_threshold:
        return "escalation"

    # Map intent to specialist node
    routing_map = {
        "billing": "billing",
        "technical": "technical",
        "escalation": "escalation",
        "account": "technical",  # Account issues handled by technical for now
        "general": "technical",
    }

    return routing_map.get(intent, "technical")


def _check_escalation(state: AgentState) -> str:
    """Conditional edge: check if specialist response needs escalation."""
    if state.get("requires_escalation", False):
        return "escalation"
    return "end"


def build_graph() -> StateGraph:
    """Build the customer service state graph.

    Graph topology:
        START -> crm_enrichment -> router -> [billing | technical | escalation]
                                                  |            |
                                                  v            v
                                          check_escalation  check_escalation
                                                  |            |
                                          [end | escalation]  [end | escalation]
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("crm_enrichment", crm_enrichment_node)
    graph.add_node("router", router_node)
    graph.add_node("billing", billing_node)
    graph.add_node("technical", technical_node)
    graph.add_node("escalation", escalation_node)

    # Set entry point
    graph.set_entry_point("crm_enrichment")

    # Linear edge: enrichment -> router
    graph.add_edge("crm_enrichment", "router")

    # Conditional edge: router -> specialist (based on intent classification)
    graph.add_conditional_edges(
        "router",
        _route_after_classification,
        {
            "billing": "billing",
            "technical": "technical",
            "escalation": "escalation",
        },
    )

    # Conditional edges: specialist -> end or escalation
    graph.add_conditional_edges(
        "billing",
        _check_escalation,
        {"end": END, "escalation": "escalation"},
    )
    graph.add_conditional_edges(
        "technical",
        _check_escalation,
        {"end": END, "escalation": "escalation"},
    )

    # Escalation always terminates the graph
    graph.add_edge("escalation", END)

    return graph


def compile_graph(enable_checkpointing: bool = True):
    """Compile the graph with optional checkpointing for human-in-the-loop.

    Args:
        enable_checkpointing: If True, enables state persistence for
            pause/resume workflows. Required for human-in-the-loop.

    Returns:
        Compiled graph ready for invocation.
    """
    graph = build_graph()

    if enable_checkpointing:
        checkpointer = MemorySaver()
        return graph.compile(checkpointer=checkpointer)

    return graph.compile()


def run_query(
    customer_id: str,
    query_text: str,
    channel: str = "chat",
    thread_id: str | None = None,
) -> AgentState:
    """Execute a customer query through the compiled graph.

    Args:
        customer_id: Customer identifier for CRM lookup.
        query_text: The customer's question or issue.
        channel: Communication channel (chat, email, phone).
        thread_id: Optional thread ID for conversation continuity.

    Returns:
        Final state after graph execution with all accumulated context.
    """
    compiled = compile_graph(enable_checkpointing=thread_id is not None)

    initial_state: AgentState = {
        "customer_id": customer_id,
        "query_text": query_text,
        "channel": channel,
        "intent": "",
        "confidence": 0.0,
        "routing_reasoning": "",
        "customer_record": None,
        "billing_history": None,
        "response_text": "",
        "actions_taken": [],
        "requires_escalation": False,
        "escalation_priority": None,
        "escalation_team": None,
        "escalation_reason": None,
        "messages": [],
        "current_node": "start",
        "iteration_count": 0,
    }

    config = {}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}

    final_state = compiled.invoke(initial_state, config=config)
    return final_state


def main() -> None:
    """Demo: run a sample query through the LangGraph pipeline."""
    result = run_query(
        customer_id="CUST-001",
        query_text="I was charged twice for last month's invoice and I need a refund immediately.",
        channel="chat",
    )

    print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
    print(f"Response: {result['response_text'][:200]}...")
    if result.get("escalation_team"):
        print(f"Escalated to: {result['escalation_team']} ({result['escalation_priority']})")


if __name__ == "__main__":
    main()
