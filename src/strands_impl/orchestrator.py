"""Main orchestrator using Strands Agents SDK.

This is the top-level entry point for the Strands implementation. It coordinates
the routing agent, specialist agents, and escalation handler in a clean pipeline.

Why Strands for this pattern:
- Native @tool decorator makes tool sharing between agents trivial
- Agent() constructor is minimal - no boilerplate graph definitions
- Direct agent invocation (agent(prompt)) feels like calling a function
- BedrockModel integration is first-class, no adapter layers
- Best choice when you want agent autonomy with lightweight orchestration
"""

from __future__ import annotations

import logging
from typing import Any

from src.common.config import get_config
from src.common.models import AgentResponse, CustomerQuery, EscalationRequest, Intent
from src.strands_impl.agents.billing import handle_billing_query
from src.strands_impl.agents.escalation import handle_escalation
from src.strands_impl.agents.router import classify_intent
from src.strands_impl.agents.technical import handle_technical_query

logger = logging.getLogger(__name__)
_config = get_config()


class StrandsOrchestrator:
    """Orchestrates multi-agent customer service using Strands Agents SDK.

    Architecture:
        Query -> Router Agent -> Specialist Agent -> Response
                                      |
                                      v (if needed)
                              Escalation Agent -> Human Handoff

    Key design decisions:
    - Each agent is stateless; conversation state is managed externally
    - Tools are shared across agents via Python imports (no registry)
    - Escalation is a first-class flow, not an afterthought
    """

    def __init__(self) -> None:
        self._conversation_memory: dict[str, list[dict[str, str]]] = {}

    def process_query(self, query: CustomerQuery) -> AgentResponse | EscalationRequest:
        """Process a customer query through the full agent pipeline.

        Args:
            query: Inbound customer query with metadata.

        Returns:
            Either a resolved AgentResponse or an EscalationRequest for human handoff.
        """
        logger.info(f"Processing query {query.id} from customer {query.customer_id}")

        # Step 1: Classify intent
        classification = classify_intent(query)
        intent = Intent(classification["intent"])
        confidence = classification.get("confidence", 0.5)

        logger.info(
            f"Classified as {intent.value} with confidence {confidence:.2f}"
        )

        # Step 2: Check if confidence is too low - route to general or escalate
        if confidence < _config.escalation_threshold:
            logger.warning(f"Low confidence ({confidence:.2f}), escalating")
            return handle_escalation(
                query=query,
                conversation_history=self._get_history(query.customer_id),
            )

        # Step 3: Route to specialist
        conversation_history = self._get_history(query.customer_id)
        response = self._route_to_specialist(intent, query, conversation_history)

        # Step 4: Record interaction in memory
        self._record_interaction(query, response)

        # Step 5: Check if specialist triggered escalation
        if response.requires_escalation and _config.enable_human_in_the_loop:
            logger.info(f"Specialist triggered escalation for query {query.id}")
            return handle_escalation(
                query=query,
                prior_response=response,
                conversation_history=self._get_history(query.customer_id),
            )

        return response

    def _route_to_specialist(
        self,
        intent: Intent,
        query: CustomerQuery,
        history: list[dict[str, str]] | None,
    ) -> AgentResponse:
        """Route to the appropriate specialist agent based on classified intent."""
        handlers: dict[Intent, Any] = {
            Intent.BILLING: handle_billing_query,
            Intent.TECHNICAL: handle_technical_query,
            Intent.ESCALATION: lambda q, h: self._force_escalation_response(q),
        }

        handler = handlers.get(intent, handle_technical_query)

        if intent == Intent.ESCALATION:
            return handler(query, history)

        return handler(query=query, conversation_history=history)

    def _force_escalation_response(self, query: CustomerQuery) -> AgentResponse:
        """Create a response that forces escalation flow."""
        return AgentResponse(
            query_id=query.id,
            agent_name="router",
            intent=Intent.ESCALATION,
            response_text="This query requires immediate human attention.",
            confidence=0.95,
            requires_escalation=True,
        )

    def _get_history(self, customer_id: str) -> list[dict[str, str]]:
        """Retrieve conversation history for a customer."""
        return self._conversation_memory.get(customer_id, [])

    def _record_interaction(
        self, query: CustomerQuery, response: AgentResponse
    ) -> None:
        """Record the interaction in conversation memory."""
        if not _config.enable_conversation_memory:
            return

        if query.customer_id not in self._conversation_memory:
            self._conversation_memory[query.customer_id] = []

        self._conversation_memory[query.customer_id].extend([
            {"role": "customer", "content": query.query_text},
            {"role": "agent", "content": response.response_text},
        ])

        # Keep only last 20 messages per customer
        self._conversation_memory[query.customer_id] = self._conversation_memory[
            query.customer_id
        ][-20:]


def main() -> None:
    """Demo: run a sample query through the orchestrator."""
    orchestrator = StrandsOrchestrator()

    sample_query = CustomerQuery(
        customer_id="CUST-001",
        query_text="I was charged twice for last month's invoice and I need a refund immediately.",
        channel="chat",
    )

    result = orchestrator.process_query(sample_query)

    if isinstance(result, AgentResponse):
        print(f"Agent: {result.agent_name}")
        print(f"Intent: {result.intent.value}")
        print(f"Response: {result.response_text}")
    elif isinstance(result, EscalationRequest):
        print(f"Escalated to: {result.assigned_team}")
        print(f"Priority: {result.priority.value}")
        print(f"Reason: {result.reason}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
