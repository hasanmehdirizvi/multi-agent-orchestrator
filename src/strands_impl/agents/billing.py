"""Billing inquiry specialist agent using Strands Agents SDK.

Handles payment disputes, invoice questions, plan changes, and refund requests.
Demonstrates tool integration with CRM and ticketing systems.
"""

from __future__ import annotations

from strands import Agent
from strands.models.bedrock import BedrockModel

from src.common.config import get_config
from src.common.models import AgentResponse, CustomerQuery, Intent
from src.strands_impl.tools.crm_lookup import crm_get_billing_history, crm_lookup
from src.strands_impl.tools.ticket_create import create_ticket, get_open_tickets

_config = get_config()

BILLING_SYSTEM_PROMPT = """You are a billing specialist agent for an enterprise customer service platform.

You handle:
- Invoice and payment inquiries
- Plan upgrades/downgrades
- Refund requests (up to $500 auto-approved, above requires escalation)
- Billing dispute resolution
- Payment method updates

Guidelines:
1. Always look up the customer record first to understand their account context.
2. Check billing history before making any changes.
3. For enterprise customers (monthly spend > $10,000), apply white-glove service.
4. Auto-approve refunds under $500 for accounts in good standing.
5. Escalate if the customer mentions legal action or regulatory complaints.
6. Create a ticket for any billing changes that require backend processing.

Respond with a clear resolution or next steps. Be professional and empathetic.
"""


def create_billing_agent() -> Agent:
    """Create the billing specialist agent with tools."""
    model = BedrockModel(
        model_id=_config.specialist_model.model_id,
        region_name=_config.specialist_model.region,
        temperature=_config.specialist_model.temperature,
        max_tokens=_config.specialist_model.max_tokens,
    )

    return Agent(
        model=model,
        system_prompt=BILLING_SYSTEM_PROMPT,
        tools=[crm_lookup, crm_get_billing_history, create_ticket, get_open_tickets],
    )


def handle_billing_query(
    query: CustomerQuery,
    conversation_history: list[dict[str, str]] | None = None,
) -> AgentResponse:
    """Process a billing-related customer query.

    Args:
        query: The classified customer query.
        conversation_history: Prior conversation context for continuity.

    Returns:
        Structured agent response with resolution or next steps.
    """
    agent = create_billing_agent()

    context_block = ""
    if conversation_history:
        history_text = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in conversation_history
        )
        context_block = f"\n\nConversation history:\n{history_text}"

    prompt = f"""Handle this billing inquiry:

Customer ID: {query.customer_id}
Query: {query.query_text}
Channel: {query.channel}
{context_block}

First look up the customer, then check their billing history, and provide a resolution.
"""

    result = agent(prompt)
    response_text = str(result)

    # Determine if escalation is needed
    requires_escalation = any(
        trigger in query.query_text.lower()
        for trigger in ["legal", "lawyer", "lawsuit", "regulator", "cancel everything"]
    )

    return AgentResponse(
        query_id=query.id,
        agent_name="billing_specialist",
        intent=Intent.BILLING,
        response_text=response_text,
        confidence=0.85,
        actions_taken=["crm_lookup", "billing_history_check"],
        requires_escalation=requires_escalation,
        context={"customer_id": query.customer_id},
    )
