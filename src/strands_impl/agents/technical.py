"""Technical support specialist agent using Strands Agents SDK.

Handles system errors, API issues, integration problems, and performance concerns.
Demonstrates multi-tool usage and structured diagnostic workflows.
"""

from __future__ import annotations

from strands import Agent
from strands.models.bedrock import BedrockModel

from src.common.config import get_config
from src.common.models import AgentResponse, CustomerQuery, Intent
from src.strands_impl.tools.crm_lookup import crm_lookup
from src.strands_impl.tools.ticket_create import create_ticket, update_ticket_status

_config = get_config()

TECHNICAL_SYSTEM_PROMPT = """You are a technical support specialist for an enterprise platform.

You handle:
- API errors and integration failures
- System performance degradation
- Authentication and authorization issues
- Data synchronization problems
- Infrastructure and connectivity issues

Diagnostic workflow:
1. Look up the customer to understand their plan tier and environment.
2. Identify the technical domain (API, auth, performance, data, infra).
3. Gather relevant error codes, timestamps, and affected services.
4. Provide immediate mitigation steps if possible.
5. Create a ticket for issues requiring engineering investigation.
6. Escalate P1/P2 issues affecting production workloads immediately.

Technical response format:
- Issue Summary: <brief description>
- Root Cause (if identifiable): <explanation>
- Immediate Steps: <what customer can do now>
- Resolution Timeline: <expected fix time>
- Ticket: <ticket ID if created>
"""


def create_technical_agent() -> Agent:
    """Create the technical support agent with diagnostic tools."""
    model = BedrockModel(
        model_id=_config.specialist_model.model_id,
        region_name=_config.specialist_model.region,
        temperature=_config.specialist_model.temperature,
        max_tokens=_config.specialist_model.max_tokens,
    )

    return Agent(
        model=model,
        system_prompt=TECHNICAL_SYSTEM_PROMPT,
        tools=[crm_lookup, create_ticket, update_ticket_status],
    )


def handle_technical_query(
    query: CustomerQuery,
    conversation_history: list[dict[str, str]] | None = None,
) -> AgentResponse:
    """Process a technical support query with diagnostic workflow.

    Args:
        query: The classified customer query.
        conversation_history: Prior conversation context.

    Returns:
        Structured agent response with diagnosis and resolution steps.
    """
    agent = create_technical_agent()

    context_block = ""
    if conversation_history:
        history_text = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in conversation_history
        )
        context_block = f"\n\nConversation history:\n{history_text}"

    prompt = f"""Diagnose and resolve this technical issue:

Customer ID: {query.customer_id}
Query: {query.query_text}
Channel: {query.channel}
Metadata: {query.metadata}
{context_block}

Follow the diagnostic workflow. Look up the customer first, then diagnose and resolve.
"""

    result = agent(prompt)
    response_text = str(result)

    # Check for production-impacting issues that need escalation
    production_keywords = ["outage", "down", "production", "p1", "critical", "data loss"]
    requires_escalation = any(
        keyword in query.query_text.lower() for keyword in production_keywords
    )

    return AgentResponse(
        query_id=query.id,
        agent_name="technical_specialist",
        intent=Intent.TECHNICAL,
        response_text=response_text,
        confidence=0.80,
        actions_taken=["crm_lookup", "diagnostic_analysis", "ticket_created"],
        requires_escalation=requires_escalation,
        context={
            "customer_id": query.customer_id,
            "severity": "high" if requires_escalation else "medium",
        },
    )
