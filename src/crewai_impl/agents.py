"""CrewAI Agent definitions for the customer service crew.

CrewAI models agents as role-playing personas with goals, backstories, and tools.
This is ideal when you want:
- Agents that reason about their identity and objectives
- Hierarchical delegation (manager -> specialist)
- Collaborative problem-solving between agents
- Natural language task decomposition

Trade-off vs Strands/LangGraph: Less control over exact execution path,
but more natural collaboration patterns. Best for complex multi-step
workflows where agent autonomy improves outcomes.
"""

from __future__ import annotations

from crewai import Agent, LLM
from crewai.tools import tool

from src.common.config import get_config

_config = get_config()


# ---------------------------------------------------------------------------
# Tool definitions (CrewAI uses its own @tool decorator)
# ---------------------------------------------------------------------------


@tool
def lookup_customer(customer_id: str) -> str:
    """Look up customer information in the CRM system.

    Args:
        customer_id: The customer identifier (e.g., CUST-001).
    """
    from src.strands_impl.tools.crm_lookup import _CRM_DB

    record = _CRM_DB.get(customer_id)
    if record is None:
        return f"Customer {customer_id} not found in CRM."
    return str(record)


@tool
def check_billing(customer_id: str) -> str:
    """Check billing history and current balance for a customer.

    Args:
        customer_id: The customer identifier.
    """
    from src.strands_impl.tools.crm_lookup import _CRM_DB

    record = _CRM_DB.get(customer_id)
    if record is None:
        return f"No billing records for {customer_id}."
    return f"Monthly spend: ${record['monthly_spend']}, Plan: {record['plan_tier']}, Status: {record['account_status']}"


@tool
def create_support_ticket(customer_id: str, subject: str, priority: str) -> str:
    """Create a support ticket for tracking.

    Args:
        customer_id: The customer identifier.
        subject: Brief description of the issue.
        priority: Ticket priority (low, medium, high, critical).
    """
    import uuid

    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    return f"Ticket {ticket_id} created for {customer_id}: {subject} (Priority: {priority})"


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

def _get_llm() -> LLM:
    """Create LLM instance for CrewAI agents."""
    return LLM(
        model=f"bedrock/{_config.specialist_model.model_id}",
        temperature=_config.specialist_model.temperature,
    )


def create_manager_agent() -> Agent:
    """Create the service manager agent that orchestrates the crew.

    The manager uses hierarchical process to delegate to specialists,
    review their work, and synthesize final responses.
    """
    return Agent(
        role="Customer Service Manager",
        goal="Ensure every customer query is resolved efficiently with high satisfaction. "
        "Route to the right specialist, review their work, and escalate when needed.",
        backstory="You are a senior customer service manager with 15 years of experience "
        "in enterprise SaaS. You know when to delegate, when to escalate, and how "
        "to ensure quality responses. You prioritize enterprise customers and watch "
        "for signals that require immediate human intervention.",
        llm=_get_llm(),
        allow_delegation=True,
        verbose=True,
    )


def create_billing_agent() -> Agent:
    """Create the billing specialist agent."""
    return Agent(
        role="Billing Specialist",
        goal="Resolve billing inquiries accurately. Handle refunds up to $500, "
        "explain charges clearly, and process plan changes.",
        backstory="You are a billing expert who has handled thousands of enterprise "
        "billing disputes. You know the pricing tiers inside out, can spot "
        "duplicate charges instantly, and always verify with the CRM before "
        "making any commitments.",
        llm=_get_llm(),
        tools=[lookup_customer, check_billing, create_support_ticket],
        allow_delegation=False,
        verbose=True,
    )


def create_technical_agent() -> Agent:
    """Create the technical support specialist agent."""
    return Agent(
        role="Technical Support Engineer",
        goal="Diagnose and resolve technical issues. Provide clear root cause analysis, "
        "immediate workarounds, and resolution timelines.",
        backstory="You are a senior support engineer with deep knowledge of API "
        "integrations, authentication systems, and cloud infrastructure. You "
        "approach every issue methodically: reproduce, diagnose, mitigate, resolve.",
        llm=_get_llm(),
        tools=[lookup_customer, create_support_ticket],
        allow_delegation=False,
        verbose=True,
    )


def create_escalation_agent() -> Agent:
    """Create the escalation handler agent."""
    return Agent(
        role="Escalation Manager",
        goal="Prepare comprehensive handoff packages for human agents. Assess priority, "
        "assign to the correct team, and ensure no context is lost.",
        backstory="You are the bridge between AI and human support. You've seen "
        "every type of escalation and know exactly what information a human agent "
        "needs to hit the ground running. You never let a critical issue slip "
        "through without proper urgency classification.",
        llm=_get_llm(),
        tools=[lookup_customer, create_support_ticket],
        allow_delegation=False,
        verbose=True,
    )
