"""Escalation handler agent using Strands Agents SDK.

Manages the human-in-the-loop escalation workflow. When specialist agents
determine they cannot resolve an issue, or when policy triggers require it,
this agent prepares the handoff package for human agents.
"""

from __future__ import annotations

from strands import Agent
from strands.models.bedrock import BedrockModel

from src.common.config import get_config
from src.common.models import (
    AgentResponse,
    CustomerQuery,
    EscalationRequest,
    Intent,
    Priority,
)
from src.strands_impl.tools.crm_lookup import crm_lookup
from src.strands_impl.tools.ticket_create import create_ticket

_config = get_config()

ESCALATION_SYSTEM_PROMPT = """You are an escalation management agent. Your role is to prepare
comprehensive handoff packages when issues need human intervention.

Escalation triggers:
- Customer explicitly requests a human agent
- Legal or regulatory threats
- Data breach or security incidents
- SLA violations on enterprise accounts
- Repeated failed resolution attempts (3+ interactions)
- Revenue risk above $10,000/month

Your responsibilities:
1. Assess the priority level (low, medium, high, critical).
2. Summarize the issue and all attempted resolutions.
3. Identify the correct team for assignment (billing_ops, engineering_l2, security, legal, retention).
4. Prepare context package with customer history and conversation.
5. Set appropriate SLA expectations based on priority.

Priority SLA targets:
- Critical: 15 minutes response, 1 hour resolution
- High: 1 hour response, 4 hour resolution
- Medium: 4 hour response, 24 hour resolution
- Low: 24 hour response, 72 hour resolution

Output a structured escalation summary with all context needed for the human agent.
"""


def create_escalation_agent() -> Agent:
    """Create the escalation handler agent."""
    model = BedrockModel(
        model_id=_config.specialist_model.model_id,
        region_name=_config.specialist_model.region,
        temperature=0.0,  # Deterministic for escalation decisions
        max_tokens=_config.specialist_model.max_tokens,
    )

    return Agent(
        model=model,
        system_prompt=ESCALATION_SYSTEM_PROMPT,
        tools=[crm_lookup, create_ticket],
    )


def handle_escalation(
    query: CustomerQuery,
    prior_response: AgentResponse | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> EscalationRequest:
    """Process an escalation request and prepare handoff package.

    Args:
        query: The original customer query.
        prior_response: Response from the specialist that triggered escalation.
        conversation_history: Full conversation history for context.

    Returns:
        Structured escalation request ready for human agent assignment.
    """
    agent = create_escalation_agent()

    prior_context = ""
    if prior_response:
        prior_context = f"""
Previous agent: {prior_response.agent_name}
Previous response: {prior_response.response_text}
Actions taken: {', '.join(prior_response.actions_taken)}
"""

    history_block = ""
    if conversation_history:
        history_block = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in conversation_history
        )

    prompt = f"""Prepare an escalation package for this issue:

Customer ID: {query.customer_id}
Original Query: {query.query_text}
Channel: {query.channel}

{prior_context}

Conversation History:
{history_block}

Assess priority, determine the correct team, and prepare the handoff summary.
Look up the customer first to understand their account value and history.
"""

    result = agent(prompt)
    response_text = str(result)

    # Determine priority based on signals
    priority = _assess_priority(query, prior_response)

    # Determine team assignment
    team = _determine_team(query, prior_response)

    return EscalationRequest(
        query_id=query.id,
        customer_id=query.customer_id,
        priority=priority,
        reason=response_text,
        conversation_history=conversation_history or [],
        attempted_resolutions=(
            prior_response.actions_taken if prior_response else []
        ),
        assigned_team=team,
    )


def _assess_priority(
    query: CustomerQuery, prior_response: AgentResponse | None
) -> Priority:
    """Assess escalation priority based on query signals."""
    text = query.query_text.lower()

    if any(w in text for w in ["breach", "security", "data loss", "legal"]):
        return Priority.CRITICAL
    if any(w in text for w in ["outage", "down", "sla", "production"]):
        return Priority.HIGH
    if any(w in text for w in ["urgent", "asap", "frustrated"]):
        return Priority.MEDIUM
    return Priority.LOW


def _determine_team(
    query: CustomerQuery, prior_response: AgentResponse | None
) -> str:
    """Determine which team should handle the escalation."""
    text = query.query_text.lower()

    if any(w in text for w in ["legal", "lawsuit", "regulator"]):
        return "legal"
    if any(w in text for w in ["breach", "security", "unauthorized"]):
        return "security"
    if any(w in text for w in ["cancel", "churn", "leaving"]):
        return "retention"
    if prior_response and prior_response.intent == Intent.BILLING:
        return "billing_ops"
    return "engineering_l2"
