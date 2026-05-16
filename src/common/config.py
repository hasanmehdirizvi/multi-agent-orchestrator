"""Shared configuration for the multi-agent orchestrator.

Centralizes model selection, timeouts, and feature flags so all three
framework implementations share identical operational parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """LLM model configuration."""

    model_id: str = "us.anthropic.claude-sonnet-4-20250514"
    temperature: float = 0.1
    max_tokens: int = 4096
    region: str = "us-west-2"


@dataclass(frozen=True)
class OrchestratorConfig:
    """Top-level orchestrator configuration."""

    # Model settings
    router_model: ModelConfig = field(default_factory=ModelConfig)
    specialist_model: ModelConfig = field(default_factory=ModelConfig)

    # Routing thresholds
    confidence_threshold: float = 0.7
    escalation_threshold: float = 0.3

    # Timeouts (seconds)
    agent_timeout: float = 30.0
    tool_timeout: float = 10.0

    # Retry policy
    max_retries: int = 3
    retry_backoff_factor: float = 1.5

    # Feature flags
    enable_human_in_the_loop: bool = True
    enable_conversation_memory: bool = True
    enable_metrics: bool = True

    # External service endpoints
    crm_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "CRM_ENDPOINT", "https://crm.internal.example.com/api/v1"
        )
    )
    ticketing_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "TICKETING_ENDPOINT", "https://tickets.internal.example.com/api/v1"
        )
    )


def get_config() -> OrchestratorConfig:
    """Load configuration with environment variable overrides."""
    return OrchestratorConfig(
        router_model=ModelConfig(
            model_id=os.getenv("ROUTER_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514"),
            temperature=float(os.getenv("ROUTER_TEMPERATURE", "0.0")),
            region=os.getenv("AWS_REGION", "us-west-2"),
        ),
        specialist_model=ModelConfig(
            model_id=os.getenv("SPECIALIST_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514"),
            temperature=float(os.getenv("SPECIALIST_TEMPERATURE", "0.1")),
            region=os.getenv("AWS_REGION", "us-west-2"),
        ),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.7")),
        escalation_threshold=float(os.getenv("ESCALATION_THRESHOLD", "0.3")),
        enable_human_in_the_loop=os.getenv("ENABLE_HITL", "true").lower() == "true",
    )
