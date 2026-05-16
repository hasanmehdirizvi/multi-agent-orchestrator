"""CRM lookup tool using Strands @tool decorator.

Provides customer record retrieval for any agent that needs account context.
In production this would call the CRM API; here we simulate with realistic data.
"""

from __future__ import annotations

from datetime import datetime

from strands import tool

from src.common.models import CRMRecord

# Simulated CRM database for demonstration
_CRM_DB: dict[str, dict] = {
    "CUST-001": {
        "customer_id": "CUST-001",
        "name": "Acme Insurance Corp",
        "email": "support@acmeinsurance.com",
        "plan_tier": "enterprise",
        "account_status": "active",
        "monthly_spend": 24500.00,
        "open_tickets": 2,
        "last_interaction": datetime(2026, 5, 14, 9, 30),
    },
    "CUST-002": {
        "customer_id": "CUST-002",
        "name": "Regional Health Partners",
        "email": "it@regionalhp.com",
        "plan_tier": "professional",
        "account_status": "active",
        "monthly_spend": 8200.00,
        "open_tickets": 0,
        "last_interaction": datetime(2026, 5, 10, 14, 15),
    },
    "CUST-003": {
        "customer_id": "CUST-003",
        "name": "SmallBiz Insurance LLC",
        "email": "admin@smallbizins.com",
        "plan_tier": "starter",
        "account_status": "past_due",
        "monthly_spend": 450.00,
        "open_tickets": 5,
        "last_interaction": datetime(2026, 4, 28, 11, 0),
    },
}


@tool
def crm_lookup(customer_id: str) -> dict:
    """Look up a customer record in the CRM system.

    Args:
        customer_id: The unique customer identifier (e.g., CUST-001).

    Returns:
        Customer record with account details, plan info, and interaction history.
    """
    record = _CRM_DB.get(customer_id)
    if record is None:
        return {
            "error": f"Customer {customer_id} not found",
            "suggestion": "Verify the customer ID format (CUST-XXX)",
        }

    crm_record = CRMRecord(**record)
    return crm_record.model_dump(mode="json")


@tool
def crm_get_billing_history(customer_id: str, months: int = 3) -> dict:
    """Retrieve billing history for a customer.

    Args:
        customer_id: The unique customer identifier.
        months: Number of months of history to retrieve (default 3).

    Returns:
        Billing history with invoices and payment status.
    """
    record = _CRM_DB.get(customer_id)
    if record is None:
        return {"error": f"Customer {customer_id} not found"}

    # Simulated billing history
    monthly_spend = record["monthly_spend"]
    history = []
    for i in range(months):
        history.append({
            "month": f"2026-{5 - i:02d}",
            "amount": monthly_spend * (1.0 + (i * 0.02)),  # slight variance
            "status": "paid" if i > 0 else "pending",
            "invoice_id": f"INV-{customer_id}-2026{5 - i:02d}",
        })

    return {
        "customer_id": customer_id,
        "plan_tier": record["plan_tier"],
        "billing_history": history,
        "payment_method": "ACH transfer",
    }
