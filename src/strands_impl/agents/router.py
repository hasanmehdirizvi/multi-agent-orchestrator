"""Intent classification and routing agent using Strands Agents SDK.

The router is the entry point for all customer queries. It classifies intent,
determines confidence, and delegates to the appropriate specialist agent.
This demonstrates Strands' native agent-to-agent delegation pattern.
"""

from __future__ import annotations

from strands import Agent
from strands.models.bedrock import BedrockModel

from src.common.config import get_config
from src.common.models import CustomerQuery, Intent

_config = get_config()

ROUTER_SYSTEM_PROMPT = """You are an intent classification agent for an enterprise customer service system.

Your job is to analyze the customer's query and classify it into exactly one intent category:
- BILLING: Payment issues, invoice questions, plan changes, refunds, pricing
- TECHNICAL: System errors, API issues, integration problems, performance, outages
- ACCOUNT: Profile changes, access management, user provisioning, SSO
- ESCALATION: Angry customers, legal threats, data breaches, SLA violations
- GENERAL: Everything else - product info, feature requests, general questions

Respond with a JSON object containing:
{
    "intent": "<INTENT_CATEGORY>",
    "confidence": <float between 0 and 1>,
    "reasoning": "<brief explanation>",
    "key_entities": ["<extracted entities relevant to routing>"]
}

Be precise. High-value enterprise customers mentioning SLA violations or threatening
to churn should be routed to ESCALATION regardless of the surface-level topic.
"""


def create_router_agent() -> Agent:
    """Create the router agent with Bedrock model configuration."""
    model = BedrockModel(
        model_id=_config.router_model.model_id,
        region_name=_config.router_model.region,
        temperature=_config.router_model.temperature,
        max_tokens=1024,
    )

    return Agent(
        model=model,
        system_prompt=ROUTER_SYSTEM_PROMPT,
    )


def classify_intent(query: CustomerQuery) -> dict:
    """Classify a customer query into an intent category.

    Args:
        query: The inbound customer query to classify.

    Returns:
        Classification result with intent, confidence, and reasoning.
    """
    router = create_router_agent()

    prompt = f"""Classify this customer query:

Customer ID: {query.customer_id}
Channel: {query.channel}
Query: {query.query_text}
"""

    result = router(prompt)

    # Parse the structured response
    import json

    try:
        classification = json.loads(str(result))
    except json.JSONDecodeError:
        # Fallback: extract intent from unstructured response
        response_text = str(result).lower()
        if "billing" in response_text:
            intent = Intent.BILLING
        elif "technical" in response_text:
            intent = Intent.TECHNICAL
        elif "escalation" in response_text:
            intent = Intent.ESCALATION
        else:
            intent = Intent.GENERAL

        classification = {
            "intent": intent.value,
            "confidence": 0.6,
            "reasoning": "Fallback classification from unstructured response",
            "key_entities": [],
        }

    return classification
