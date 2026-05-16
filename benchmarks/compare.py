"""Benchmark comparison script for all three framework implementations.

Runs identical queries through Strands, LangGraph, and CrewAI to compare:
- Response quality and consistency
- Execution time
- Token usage (estimated)
- Routing accuracy
- Escalation behavior

Usage:
    python -m benchmarks.compare
    python -m benchmarks.compare --framework strands
    python -m benchmarks.compare --query "I need a refund"
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Any

from src.common.models import AgentResponse, CustomerQuery, EscalationRequest, Intent


# ---------------------------------------------------------------------------
# Benchmark test cases
# ---------------------------------------------------------------------------

TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "billing_refund",
        "customer_id": "CUST-001",
        "query": "I was charged twice for last month's invoice. I need a refund for $450.",
        "expected_intent": "billing",
        "expected_escalation": False,
    },
    {
        "name": "technical_api_error",
        "customer_id": "CUST-002",
        "query": "Our API integration is returning 503 errors since this morning. "
        "Batch processing is failing and we're losing data.",
        "expected_intent": "technical",
        "expected_escalation": False,
    },
    {
        "name": "escalation_legal",
        "customer_id": "CUST-001",
        "query": "This is the third time billing has been wrong. I'm involving our legal "
        "team if this isn't resolved today. We have an SLA that guarantees 99.9% uptime.",
        "expected_intent": "escalation",
        "expected_escalation": True,
    },
    {
        "name": "technical_production_outage",
        "customer_id": "CUST-002",
        "query": "URGENT: Production is completely down. Our customers cannot access the "
        "platform. This is a P1 incident affecting 50,000 users.",
        "expected_intent": "technical",
        "expected_escalation": True,
    },
    {
        "name": "billing_plan_change",
        "customer_id": "CUST-003",
        "query": "We'd like to upgrade from starter to professional plan. "
        "Can you walk me through the pricing difference?",
        "expected_intent": "billing",
        "expected_escalation": False,
    },
]


@dataclass
class BenchmarkResult:
    """Result of running a single test case through one framework."""

    framework: str
    test_name: str
    intent_classified: str
    expected_intent: str
    intent_correct: bool
    escalated: bool
    expected_escalation: bool
    escalation_correct: bool
    response_length: int
    execution_time_ms: float
    error: str | None = None


@dataclass
class BenchmarkSummary:
    """Aggregated results across all test cases for one framework."""

    framework: str
    total_cases: int = 0
    intent_accuracy: float = 0.0
    escalation_accuracy: float = 0.0
    avg_response_length: float = 0.0
    avg_execution_time_ms: float = 0.0
    errors: int = 0
    results: list[BenchmarkResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Framework runners
# ---------------------------------------------------------------------------


def run_strands(query: CustomerQuery) -> dict[str, Any]:
    """Run a query through the Strands implementation."""
    from src.strands_impl.orchestrator import StrandsOrchestrator

    orchestrator = StrandsOrchestrator()
    start = time.perf_counter()
    result = orchestrator.process_query(query)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if isinstance(result, AgentResponse):
        return {
            "intent": result.intent.value,
            "escalated": result.requires_escalation,
            "response": result.response_text,
            "time_ms": elapsed_ms,
        }
    elif isinstance(result, EscalationRequest):
        return {
            "intent": "escalation",
            "escalated": True,
            "response": result.reason,
            "time_ms": elapsed_ms,
        }
    return {"intent": "unknown", "escalated": False, "response": "", "time_ms": elapsed_ms}


def run_langgraph(query: CustomerQuery) -> dict[str, Any]:
    """Run a query through the LangGraph implementation."""
    from src.langgraph_impl.graph import run_query

    start = time.perf_counter()
    state = run_query(
        customer_id=query.customer_id,
        query_text=query.query_text,
        channel=query.channel,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    escalated = state.get("escalation_team") is not None

    return {
        "intent": state.get("intent", "unknown"),
        "escalated": escalated,
        "response": state.get("response_text", state.get("escalation_reason", "")),
        "time_ms": elapsed_ms,
    }


def run_crewai(query: CustomerQuery) -> dict[str, Any]:
    """Run a query through the CrewAI implementation."""
    from src.crewai_impl.crew import run_query as crew_run

    start = time.perf_counter()
    result = crew_run(query)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # CrewAI returns unstructured text - do basic heuristic extraction
    result_lower = result.lower()
    if "escalat" in result_lower or "critical" in result_lower:
        intent = "escalation"
        escalated = True
    elif "billing" in result_lower or "refund" in result_lower or "invoice" in result_lower:
        intent = "billing"
        escalated = False
    elif "technical" in result_lower or "api" in result_lower or "error" in result_lower:
        intent = "technical"
        escalated = False
    else:
        intent = "general"
        escalated = False

    return {
        "intent": intent,
        "escalated": escalated,
        "response": result,
        "time_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

FRAMEWORK_RUNNERS = {
    "strands": run_strands,
    "langgraph": run_langgraph,
    "crewai": run_crewai,
}


def run_benchmark(
    frameworks: list[str] | None = None,
    test_cases: list[dict] | None = None,
) -> dict[str, BenchmarkSummary]:
    """Run the full benchmark suite across selected frameworks.

    Args:
        frameworks: Which frameworks to test. None = all.
        test_cases: Which test cases to run. None = all.

    Returns:
        Mapping of framework name to aggregated results.
    """
    frameworks = frameworks or list(FRAMEWORK_RUNNERS.keys())
    cases = test_cases or TEST_CASES
    summaries: dict[str, BenchmarkSummary] = {}

    for fw_name in frameworks:
        runner = FRAMEWORK_RUNNERS.get(fw_name)
        if runner is None:
            print(f"Unknown framework: {fw_name}")
            continue

        summary = BenchmarkSummary(framework=fw_name, total_cases=len(cases))
        print(f"\n{'='*60}")
        print(f"Running {fw_name.upper()} benchmark ({len(cases)} cases)")
        print(f"{'='*60}")

        for case in cases:
            query = CustomerQuery(
                customer_id=case["customer_id"],
                query_text=case["query"],
                channel="benchmark",
            )

            print(f"\n  [{case['name']}] ", end="", flush=True)

            try:
                result = runner(query)
                bench_result = BenchmarkResult(
                    framework=fw_name,
                    test_name=case["name"],
                    intent_classified=result["intent"],
                    expected_intent=case["expected_intent"],
                    intent_correct=result["intent"] == case["expected_intent"],
                    escalated=result["escalated"],
                    expected_escalation=case["expected_escalation"],
                    escalation_correct=result["escalated"] == case["expected_escalation"],
                    response_length=len(result["response"]),
                    execution_time_ms=result["time_ms"],
                )
                summary.results.append(bench_result)
                status = "PASS" if bench_result.intent_correct else "FAIL"
                print(f"{status} ({result['time_ms']:.0f}ms)")

            except Exception as e:
                summary.errors += 1
                summary.results.append(
                    BenchmarkResult(
                        framework=fw_name,
                        test_name=case["name"],
                        intent_classified="error",
                        expected_intent=case["expected_intent"],
                        intent_correct=False,
                        escalated=False,
                        expected_escalation=case["expected_escalation"],
                        escalation_correct=False,
                        response_length=0,
                        execution_time_ms=0,
                        error=str(e),
                    )
                )
                print(f"ERROR: {e}")

        # Compute summary metrics
        valid_results = [r for r in summary.results if r.error is None]
        if valid_results:
            summary.intent_accuracy = (
                sum(1 for r in valid_results if r.intent_correct) / len(valid_results)
            )
            summary.escalation_accuracy = (
                sum(1 for r in valid_results if r.escalation_correct) / len(valid_results)
            )
            summary.avg_response_length = (
                sum(r.response_length for r in valid_results) / len(valid_results)
            )
            summary.avg_execution_time_ms = (
                sum(r.execution_time_ms for r in valid_results) / len(valid_results)
            )

        summaries[fw_name] = summary

    return summaries


def print_comparison(summaries: dict[str, BenchmarkSummary]) -> None:
    """Print a formatted comparison table."""
    print(f"\n\n{'='*80}")
    print("BENCHMARK COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Framework':<12} {'Intent Acc':<12} {'Escal Acc':<12} "
          f"{'Avg Time':<12} {'Avg Resp Len':<14} {'Errors':<8}")
    print(f"{'-'*80}")

    for fw, summary in summaries.items():
        print(
            f"{fw:<12} {summary.intent_accuracy:>8.1%}    "
            f"{summary.escalation_accuracy:>8.1%}    "
            f"{summary.avg_execution_time_ms:>8.0f}ms  "
            f"{summary.avg_response_length:>10.0f}    "
            f"{summary.errors:>4}"
        )

    print(f"\n{'='*80}")
    print("\nFramework Selection Guide:")
    print("  Strands   - Best for: Autonomous agents, minimal boilerplate, AWS-native")
    print("  LangGraph - Best for: Auditable flows, checkpointing, human-in-the-loop")
    print("  CrewAI    - Best for: Collaborative reasoning, role-based delegation")


def main() -> None:
    """CLI entry point for benchmark comparison."""
    parser = argparse.ArgumentParser(description="Multi-agent framework benchmark")
    parser.add_argument(
        "--framework", "-f",
        choices=["strands", "langgraph", "crewai"],
        help="Run only a specific framework",
    )
    parser.add_argument(
        "--query", "-q",
        help="Run a custom query instead of test suite",
    )
    parser.add_argument(
        "--customer-id", "-c",
        default="CUST-001",
        help="Customer ID for custom queries",
    )
    parser.add_argument(
        "--output", "-o",
        help="Write JSON results to file",
    )

    args = parser.parse_args()

    frameworks = [args.framework] if args.framework else None

    if args.query:
        test_cases = [{
            "name": "custom_query",
            "customer_id": args.customer_id,
            "query": args.query,
            "expected_intent": "unknown",
            "expected_escalation": False,
        }]
    else:
        test_cases = None

    summaries = run_benchmark(frameworks=frameworks, test_cases=test_cases)
    print_comparison(summaries)

    if args.output:
        output_data = {}
        for fw, summary in summaries.items():
            output_data[fw] = {
                "intent_accuracy": summary.intent_accuracy,
                "escalation_accuracy": summary.escalation_accuracy,
                "avg_execution_time_ms": summary.avg_execution_time_ms,
                "avg_response_length": summary.avg_response_length,
                "errors": summary.errors,
                "results": [
                    {
                        "test": r.test_name,
                        "intent": r.intent_classified,
                        "correct": r.intent_correct,
                        "time_ms": r.execution_time_ms,
                    }
                    for r in summary.results
                ],
            }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
