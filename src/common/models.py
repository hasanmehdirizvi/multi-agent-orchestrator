"""Shared Pydantic models for the multi-agent orchestrator.

These models define the contract between agents regardless of which
framework implementation is executing.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Intent(str, Enum):
    """Classified intent categories for customer queries."""

    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    ESCALATION = "escalation"
    GENERAL = "general"


class Priority(str, Enum):
    """Escalation priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CustomerQuery(BaseModel):
    """Inbound customer query with metadata."""

    id: UUID = Field(default_factory=uuid4)
    customer_id: str
    query_text: str
    channel: str = "chat"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Standardized response from any specialist agent."""

    query_id: UUID
    agent_name: str
    intent: Intent
    response_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    actions_taken: list[str] = Field(default_factory=list)
    requires_escalation: bool = False
    context: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EscalationRequest(BaseModel):
    """Request to escalate to human agent with full context."""

    query_id: UUID
    customer_id: str
    priority: Priority
    reason: str
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    attempted_resolutions: list[str] = Field(default_factory=list)
    assigned_team: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CRMRecord(BaseModel):
    """Customer record from CRM system."""

    customer_id: str
    name: str
    email: str
    plan_tier: str
    account_status: str
    monthly_spend: float
    open_tickets: int = 0
    last_interaction: datetime | None = None


class TicketRecord(BaseModel):
    """Support ticket record."""

    ticket_id: str
    customer_id: str
    subject: str
    description: str
    priority: Priority
    status: str = "open"
    assigned_agent: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
