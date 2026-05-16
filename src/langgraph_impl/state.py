"""LangGraph state definition using TypedDict.

Defines the shared state that flows through the graph. Every node reads from
and writes to this state, enabling full observability and replay.

Why LangGraph for stateful orchestration:
- TypedDict state gives you compile-time guarantees on data flow
- Every state transition is logged and replayable
- Conditional edges make routing logic explicit and testable
- Built-in checkpointing enables pause/resume for human-in-the-loop
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state flowing through the customer service graph.

    Each field represents a piece of context that accumulates as the query
    moves through routing, specialist processing, and potential escalation.
    """

    # Core query fields
    customer_id: str
    query_text: str
    channel: str

    # Router output
    intent: str
    confidence: float
    routing_reasoning: str

    # CRM context (populated by tools)
    customer_record: dict | None
    billing_history: list[dict] | None

    # Specialist output
    response_text: str
    actions_taken: list[str]
    requires_escalation: bool

    # Escalation fields
    escalation_priority: str | None
    escalation_team: str | None
    escalation_reason: str | None

    # Conversation tracking
    messages: Annotated[list, add_messages]

    # Metadata
    current_node: str
    iteration_count: int
