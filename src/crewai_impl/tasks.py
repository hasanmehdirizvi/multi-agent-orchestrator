"""CrewAI Task definitions for the customer service workflow.

Tasks define the work units that agents execute. In CrewAI's model:
- Tasks have descriptions, expected outputs, and assigned agents
- Tasks can depend on other tasks (sequential flow)
- The manager can reassign tasks dynamically in hierarchical mode
"""

from __future__ import annotations

from crewai import Agent, Task


def create_classification_task(query_text: str, customer_id: str, manager: Agent) -> Task:
    """Task: Classify the customer query intent.

    The manager handles classification directly since it determines delegation.
    """
    return Task(
        description=f"""Classify this customer query into one category:
- BILLING: Payment, invoice, pricing, refund, plan changes
- TECHNICAL: API errors, system issues, integration, performance
- ESCALATION: Legal threats, SLA violations, security incidents, angry customer
- GENERAL: Product questions, feature requests, other

Customer ID: {customer_id}
Query: {query_text}

Determine the intent, your confidence level (0-1), and brief reasoning.
If confidence is below 0.5, default to escalation.""",
        expected_output="Intent classification with category, confidence score, and routing decision.",
        agent=manager,
    )


def create_billing_resolution_task(
    query_text: str, customer_id: str, billing_agent: Agent
) -> Task:
    """Task: Resolve a billing inquiry."""
    return Task(
        description=f"""Resolve this billing inquiry for customer {customer_id}:

Query: {query_text}

Steps:
1. Look up the customer in CRM to understand their account.
2. Check their billing history for anomalies.
3. Provide a clear resolution or explanation.
4. If a refund is needed and <= $500, approve it.
5. If > $500 or complex, create a ticket for billing ops.

Be specific about amounts, dates, and next steps.""",
        expected_output="Detailed resolution with specific actions taken, amounts, and any ticket IDs created.",
        agent=billing_agent,
    )


def create_technical_resolution_task(
    query_text: str, customer_id: str, technical_agent: Agent
) -> Task:
    """Task: Diagnose and resolve a technical issue."""
    return Task(
        description=f"""Diagnose and resolve this technical issue for customer {customer_id}:

Query: {query_text}

Diagnostic workflow:
1. Look up customer to understand their environment and plan tier.
2. Identify the technical domain (API, auth, performance, data, infra).
3. Determine likely root cause.
4. Provide immediate mitigation steps.
5. Create a ticket if engineering investigation is needed.

Format your response as:
- Issue Summary
- Root Cause Analysis
- Immediate Steps
- Resolution Timeline
- Ticket ID (if created)""",
        expected_output="Structured diagnostic report with root cause, mitigation steps, and resolution timeline.",
        agent=technical_agent,
    )


def create_escalation_task(
    query_text: str,
    customer_id: str,
    escalation_agent: Agent,
    prior_context: str = "",
) -> Task:
    """Task: Prepare escalation package for human handoff."""
    return Task(
        description=f"""Prepare an escalation package for human agent handoff.

Customer ID: {customer_id}
Original Query: {query_text}
Prior Context: {prior_context or 'First contact - no prior resolution attempts.'}

Required outputs:
1. Priority assessment (critical/high/medium/low)
2. Assigned team (security, legal, engineering_l2, retention, billing_ops)
3. Situation summary (what happened, what was tried)
4. Recommended next steps for the human agent
5. SLA target based on priority

Priority guidelines:
- Critical: Security breach, data loss, legal threats -> 15 min response
- High: Production outage, SLA violation -> 1 hour response
- Medium: Unresolved after multiple attempts, upset customer -> 4 hour response
- Low: General escalation request -> 24 hour response""",
        expected_output="Complete escalation package with priority, team assignment, summary, and SLA target.",
        agent=escalation_agent,
    )
