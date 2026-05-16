"""CrewAI Crew with hierarchical process for customer service.

This module assembles agents and tasks into an executable crew.
CrewAI's hierarchical process means:
- A manager agent oversees the workflow
- It can delegate tasks to specialists dynamically
- It reviews specialist output before finalizing
- It can reassign or escalate based on quality

Why CrewAI for this pattern:
- Natural hierarchical delegation (manager/specialist)
- Agents reason about their roles and collaborate
- Built-in retry and quality review by the manager
- Best for workflows where agent judgment matters more than deterministic paths

Trade-off: Less predictable execution paths, harder to audit exact decisions.
Use when you trust agent reasoning over rigid graph topology.
"""

from __future__ import annotations

from crewai import Crew, Process

from src.common.config import get_config
from src.common.models import CustomerQuery, Intent
from src.crewai_impl.agents import (
    create_billing_agent,
    create_escalation_agent,
    create_manager_agent,
    create_technical_agent,
)
from src.crewai_impl.tasks import (
    create_billing_resolution_task,
    create_classification_task,
    create_escalation_task,
    create_technical_resolution_task,
)

_config = get_config()


def build_crew(query: CustomerQuery) -> Crew:
    """Build a customer service crew for a specific query.

    The crew composition adapts based on the query - we always include
    the manager and relevant specialists. This keeps execution efficient
    by not loading unnecessary agents.

    Args:
        query: The customer query to process.

    Returns:
        Configured Crew ready for kickoff.
    """
    # Create agents
    manager = create_manager_agent()
    billing_agent = create_billing_agent()
    technical_agent = create_technical_agent()
    escalation_agent = create_escalation_agent()

    # Create the classification task (always first)
    classification_task = create_classification_task(
        query_text=query.query_text,
        customer_id=query.customer_id,
        manager=manager,
    )

    # Create all possible resolution tasks
    billing_task = create_billing_resolution_task(
        query_text=query.query_text,
        customer_id=query.customer_id,
        billing_agent=billing_agent,
    )

    technical_task = create_technical_resolution_task(
        query_text=query.query_text,
        customer_id=query.customer_id,
        technical_agent=technical_agent,
    )

    escalation_task = create_escalation_task(
        query_text=query.query_text,
        customer_id=query.customer_id,
        escalation_agent=escalation_agent,
    )

    # In hierarchical mode, the manager decides which tasks to delegate
    # We include all tasks - the manager will select based on classification
    crew = Crew(
        agents=[manager, billing_agent, technical_agent, escalation_agent],
        tasks=[classification_task, billing_task, technical_task, escalation_task],
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=True,
    )

    return crew


def build_targeted_crew(query: CustomerQuery, intent: Intent) -> Crew:
    """Build a crew targeted to a specific intent for faster execution.

    Use this when you have already classified the intent externally
    (e.g., from the benchmark runner) and want to skip the classification step.

    Args:
        query: The customer query to process.
        intent: Pre-classified intent.

    Returns:
        Crew with only the relevant specialist.
    """
    manager = create_manager_agent()

    if intent == Intent.BILLING:
        specialist = create_billing_agent()
        task = create_billing_resolution_task(
            query_text=query.query_text,
            customer_id=query.customer_id,
            billing_agent=specialist,
        )
    elif intent == Intent.TECHNICAL:
        specialist = create_technical_agent()
        task = create_technical_resolution_task(
            query_text=query.query_text,
            customer_id=query.customer_id,
            technical_agent=specialist,
        )
    else:
        specialist = create_escalation_agent()
        task = create_escalation_task(
            query_text=query.query_text,
            customer_id=query.customer_id,
            escalation_agent=specialist,
        )

    return Crew(
        agents=[manager, specialist],
        tasks=[task],
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=True,
    )


def run_query(query: CustomerQuery) -> str:
    """Execute a customer query through the CrewAI pipeline.

    Args:
        query: The customer query to process.

    Returns:
        Final crew output as a string.
    """
    crew = build_crew(query)
    result = crew.kickoff()
    return str(result)


def main() -> None:
    """Demo: run a sample query through the CrewAI crew."""
    sample_query = CustomerQuery(
        customer_id="CUST-001",
        query_text="I was charged twice for last month's invoice and I need a refund immediately.",
        channel="chat",
    )

    result = run_query(sample_query)
    print(f"Crew Result:\n{result}")


if __name__ == "__main__":
    main()
