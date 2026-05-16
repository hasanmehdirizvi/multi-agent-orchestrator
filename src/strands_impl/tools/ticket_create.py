"""Ticket creation tool using Strands @tool decorator.

Creates support tickets in the ticketing system with proper categorization
and priority assignment. Shared across all specialist agents.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from strands import tool

from src.common.models import Priority, TicketRecord

# In-memory ticket store for demonstration
_TICKET_STORE: dict[str, dict] = {}


@tool
def create_ticket(
    customer_id: str,
    subject: str,
    description: str,
    priority: str = "medium",
    assigned_agent: str = "",
) -> dict:
    """Create a new support ticket in the ticketing system.

    Args:
        customer_id: The customer who raised the issue.
        subject: Brief summary of the issue.
        description: Detailed description of the problem.
        priority: Ticket priority - low, medium, high, or critical.
        assigned_agent: Agent or team to assign the ticket to.

    Returns:
        Created ticket record with ticket ID and status.
    """
    ticket_id = f"TKT-{uuid4().hex[:8].upper()}"

    try:
        priority_enum = Priority(priority.lower())
    except ValueError:
        priority_enum = Priority.MEDIUM

    ticket = TicketRecord(
        ticket_id=ticket_id,
        customer_id=customer_id,
        subject=subject,
        description=description,
        priority=priority_enum,
        status="open",
        assigned_agent=assigned_agent or None,
        created_at=datetime.utcnow(),
    )

    _TICKET_STORE[ticket_id] = ticket.model_dump(mode="json")

    return {
        "ticket_id": ticket_id,
        "status": "created",
        "priority": priority_enum.value,
        "message": f"Ticket {ticket_id} created successfully",
    }


@tool
def update_ticket_status(ticket_id: str, status: str, notes: str = "") -> dict:
    """Update the status of an existing support ticket.

    Args:
        ticket_id: The ticket ID to update.
        status: New status - open, in_progress, waiting_customer, resolved, closed.
        notes: Optional notes about the status change.

    Returns:
        Updated ticket status confirmation.
    """
    if ticket_id not in _TICKET_STORE:
        return {"error": f"Ticket {ticket_id} not found"}

    valid_statuses = {"open", "in_progress", "waiting_customer", "resolved", "closed"}
    if status not in valid_statuses:
        return {"error": f"Invalid status. Must be one of: {valid_statuses}"}

    _TICKET_STORE[ticket_id]["status"] = status
    return {
        "ticket_id": ticket_id,
        "new_status": status,
        "notes": notes,
        "updated_at": datetime.utcnow().isoformat(),
    }


@tool
def get_open_tickets(customer_id: str) -> dict:
    """Get all open tickets for a customer.

    Args:
        customer_id: The customer ID to look up tickets for.

    Returns:
        List of open tickets for the customer.
    """
    open_tickets = [
        ticket
        for ticket in _TICKET_STORE.values()
        if ticket["customer_id"] == customer_id and ticket["status"] in ("open", "in_progress")
    ]

    return {
        "customer_id": customer_id,
        "open_ticket_count": len(open_tickets),
        "tickets": open_tickets,
    }
