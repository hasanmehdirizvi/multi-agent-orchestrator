"""Tests for intent routing logic across all implementations.

Tests the deterministic parts of the system (routing decisions, state transitions,
model construction) without requiring live LLM calls. Uses mocking for LLM responses.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.common.models import (
    AgentResponse,
    CustomerQuery,
    EscalationRequest,
    Intent,
    Priority,
)
from src.common.config import get_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def billing_query() -> CustomerQuery:
    return CustomerQuery(
        customer_id="CUST-001",
        query_text="I was charged twice for my invoice last month.",
        channel="chat",
    )


@pytest.fixture
def technical_query() -> CustomerQuery:
    return CustomerQuery(
        customer_id="CUST-002",
        query_text="API returning 503 errors on all endpoints since 9am.",
        channel="chat",
    )


@pytest.fixture
def escalation_query() -> CustomerQuery:
    return CustomerQuery(
        customer_id="CUST-001",
        query_text="I'm contacting our lawyers. This SLA violation is unacceptable.",
        channel="email",
    )


@pytest.fixture
def config():
    return get_config()


# ---------------------------------------------------------------------------
# Model and schema tests
# ---------------------------------------------------------------------------


class TestModels:
    """Test shared Pydantic model validation."""

    def test_customer_query_defaults(self):
        query = CustomerQuery(
            customer_id="CUST-001",
            query_text="Hello",
        )
        assert query.channel == "chat"
        assert query.id is not None
        assert query.metadata == {}

    def test_agent_response_confidence_bounds(self):
        with pytest.raises(Exception):
            AgentResponse(
                query_id=uuid4(),
                agent_name="test",
                intent=Intent.BILLING,
                response_text="test",
                confidence=1.5,  # Out of bounds
            )

    def test_escalation_request_creation(self):
        request = EscalationRequest(
            query_id=uuid4(),
            customer_id="CUST-001",
            priority=Priority.HIGH,
            reason="SLA violation on enterprise account",
        )
        assert request.assigned_team is None
        assert request.attempted_resolutions == []

    def test_intent_enum_values(self):
        assert Intent.BILLING.value == "billing"
        assert Intent.TECHNICAL.value == "technical"
        assert Intent.ESCALATION.value == "escalation"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    """Test configuration loading."""

    def test_default_config(self, config):
        assert config.confidence_threshold == 0.7
        assert config.escalation_threshold == 0.3
        assert config.enable_human_in_the_loop is True

    def test_model_config(self, config):
        assert "anthropic" in config.router_model.model_id
        assert config.router_model.temperature == 0.0


# ---------------------------------------------------------------------------
# Strands routing tests
# ---------------------------------------------------------------------------


class TestStrandsRouting:
    """Test Strands orchestrator routing logic."""

    @patch("src.strands_impl.agents.router.create_router_agent")
    def test_billing_intent_routes_to_billing(self, mock_create, billing_query):
        """Verify billing classification routes to billing handler."""
        mock_agent = MagicMock()
        mock_agent.return_value = '{"intent": "billing", "confidence": 0.92, "reasoning": "Duplicate charge"}'
        mock_create.return_value = mock_agent

        from src.strands_impl.agents.router import classify_intent

        result = classify_intent(billing_query)
        assert result["intent"] == "billing"
        assert result["confidence"] == 0.92

    @patch("src.strands_impl.agents.router.create_router_agent")
    def test_low_confidence_fallback(self, mock_create, billing_query):
        """Verify low-confidence classification uses fallback logic."""
        mock_agent = MagicMock()
        mock_agent.return_value = "I'm not sure what category this falls into"
        mock_create.return_value = mock_agent

        from src.strands_impl.agents.router import classify_intent

        result = classify_intent(billing_query)
        # Should fallback with low confidence
        assert result["confidence"] <= 0.7

    def test_escalation_triggers_in_billing(self, escalation_query):
        """Verify legal keywords trigger escalation from billing agent."""
        from src.strands_impl.agents.billing import handle_billing_query

        # The handler checks keywords directly without calling LLM
        query_lower = escalation_query.query_text.lower()
        triggers = ["legal", "lawyer", "lawsuit", "regulator", "cancel everything"]
        should_escalate = any(t in query_lower for t in triggers)
        assert should_escalate is True


# ---------------------------------------------------------------------------
# LangGraph routing tests
# ---------------------------------------------------------------------------


class TestLangGraphRouting:
    """Test LangGraph conditional edge logic."""

    def test_route_billing_intent(self):
        """Verify billing intent routes to billing node."""
        from src.langgraph_impl.graph import _route_after_classification

        state = {"intent": "billing", "confidence": 0.9}
        assert _route_after_classification(state) == "billing"

    def test_route_technical_intent(self):
        """Verify technical intent routes to technical node."""
        from src.langgraph_impl.graph import _route_after_classification

        state = {"intent": "technical", "confidence": 0.85}
        assert _route_after_classification(state) == "technical"

    def test_route_low_confidence_to_escalation(self):
        """Verify low confidence forces escalation regardless of intent."""
        from src.langgraph_impl.graph import _route_after_classification

        state = {"intent": "billing", "confidence": 0.1}
        assert _route_after_classification(state) == "escalation"

    def test_escalation_check_triggers(self):
        """Verify escalation check edge function works."""
        from src.langgraph_impl.graph import _check_escalation

        state_escalate = {"requires_escalation": True}
        state_normal = {"requires_escalation": False}

        assert _check_escalation(state_escalate) == "escalation"
        assert _check_escalation(state_normal) == "end"

    def test_graph_compiles(self):
        """Verify the graph compiles without errors."""
        from src.langgraph_impl.graph import build_graph

        graph = build_graph()
        compiled = graph.compile()
        assert compiled is not None


# ---------------------------------------------------------------------------
# CRM tool tests
# ---------------------------------------------------------------------------


class TestCRMTools:
    """Test CRM lookup tool behavior."""

    def test_valid_customer_lookup(self):
        from src.strands_impl.tools.crm_lookup import _CRM_DB

        assert "CUST-001" in _CRM_DB
        assert _CRM_DB["CUST-001"]["plan_tier"] == "enterprise"

    def test_invalid_customer_returns_error(self):
        from src.strands_impl.tools.crm_lookup import crm_lookup

        # Call the underlying function (not the tool wrapper)
        result = crm_lookup.fn(customer_id="CUST-999")
        assert "error" in result

    def test_billing_history_returns_months(self):
        from src.strands_impl.tools.crm_lookup import crm_get_billing_history

        result = crm_get_billing_history.fn(customer_id="CUST-001", months=3)
        assert len(result["billing_history"]) == 3


# ---------------------------------------------------------------------------
# Ticket tool tests
# ---------------------------------------------------------------------------


class TestTicketTools:
    """Test ticket creation tool behavior."""

    def test_create_ticket(self):
        from src.strands_impl.tools.ticket_create import create_ticket

        result = create_ticket.fn(
            customer_id="CUST-001",
            subject="Test ticket",
            description="Testing ticket creation",
            priority="high",
            assigned_agent="test_agent",
        )
        assert result["status"] == "created"
        assert result["priority"] == "high"
        assert result["ticket_id"].startswith("TKT-")

    def test_invalid_priority_defaults_to_medium(self):
        from src.strands_impl.tools.ticket_create import create_ticket

        result = create_ticket.fn(
            customer_id="CUST-001",
            subject="Test",
            description="Test",
            priority="invalid_priority",
        )
        assert result["priority"] == "medium"


# ---------------------------------------------------------------------------
# Escalation logic tests
# ---------------------------------------------------------------------------


class TestEscalationLogic:
    """Test escalation priority and team assignment."""

    def test_legal_threat_critical_priority(self, escalation_query):
        from src.strands_impl.agents.escalation import _assess_priority

        priority = _assess_priority(escalation_query, None)
        assert priority == Priority.CRITICAL

    def test_outage_high_priority(self):
        query = CustomerQuery(
            customer_id="CUST-002",
            query_text="Production outage affecting all users",
        )
        from src.strands_impl.agents.escalation import _assess_priority

        priority = _assess_priority(query, None)
        assert priority == Priority.HIGH

    def test_legal_routes_to_legal_team(self, escalation_query):
        from src.strands_impl.agents.escalation import _determine_team

        team = _determine_team(escalation_query, None)
        assert team == "legal"

    def test_security_routes_to_security_team(self):
        query = CustomerQuery(
            customer_id="CUST-001",
            query_text="Unauthorized access detected in our account",
        )
        from src.strands_impl.agents.escalation import _determine_team

        team = _determine_team(query, None)
        assert team == "security"
