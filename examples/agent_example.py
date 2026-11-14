"""FlowTrace example: Mock research agent.

Demonstrates zero-config instrumentation of a realistic multi-step
LLM agent workflow using:

- @trace decorator on high-level functions (creates nested steps)
- TraceContext context manager for the top-level trace
- current_trace().record_llm_call() for prompt/response inspection
- current_trace().record_tool_call() for tool invocations
- tc.step() context manager for explicit phases

Run this file to generate a rich trace, then run ``flowtrace serve``
and open http://127.0.0.1:8080 to explore the interactive SVG graph,
prompt inspector, and tool browser.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any

from flowtrace import TraceContext, current_trace, trace
from flowtrace.models import Trace
from flowtrace.storage import SQLiteStore


@trace("Web Search")
def web_search(query: str) -> dict[str, Any]:
    """Mock web search tool.

    Records itself as a tool call so the exact arguments and output
    appear in the tool browser panel.
    """
    tc = current_trace()
    if tc:
        tc.record_tool_call(
            name="web_search",
            args={"query": query, "limit": 5},
            result={"status": "ok", "hit_count": random.randint(80, 420)},
        )

    # Simulate network latency
    time.sleep(0.25 + random.random() * 0.3)

    return {
        "query": query,
        "results": [
            f"Document A: Recent advances related to {query}",
            f"Document B: Survey paper on {query} (2024)",
            f"Document C: Case study — {query} in production",
        ],
    }


def llm_call(prompt: str, temperature: float = 0.7) -> str:
    """Mock LLM invocation that automatically records prompt, response,
    token count and latency when a trace is active.

    This is the pattern used by real agent frameworks — simply call
    this helper (or your real LLM client) inside traced functions.
    """
    tc = current_trace()
    start = time.perf_counter()

    # Simulate variable latency
    time.sleep(0.45 + random.random() * 0.55)
    latency = time.perf_counter() - start

    # Produce varied but realistic responses for the inspector
    if "synthesize" in prompt.lower() or "final answer" in prompt.lower():
        response = (
            "After reviewing the collected material, the consensus is that "
            "observability is critical for debugging agentic workflows. "
            "FlowTrace provides exactly the local, framework-agnostic view "
            "developers need."
        )
    elif "analyze" in prompt.lower() or "summarize" in prompt.lower():
        response = (
            "The documents highlight three recurring themes: latency in tool "
            "calling, prompt brittleness, and the difficulty of tracing "
            "nested LLM calls. Token usage grows super-linearly with depth."
        )
    else:
        response = (
            "Key insight: Structured tracing with clear step boundaries and "
            "rich metadata (prompts, tool inputs, durations) dramatically "
            "improves developer velocity when debugging LLM agents."
        )

    if tc:
        tc.record_llm_call(
            prompt=prompt.strip(),
            response=response,
            tokens=random.randint(180, 620),
            latency=latency,
        )

    return response


@trace("Analyze Results")
def analyze_results(query: str, search_results: dict[str, Any]) -> str:
    """Analyze search results with an LLM.

    The @trace decorator automatically wraps this function in a timed step.
    """
    prompt = f"""Analyze the following search results about "{query}".
Focus on key findings, contradictions, and suggested next steps.

Search results:
{json.dumps(search_results, indent=2)}
"""
    return llm_call(prompt)


@trace("Synthesize Answer")
def synthesize_answer(sub_answers: list[str]) -> str:
    """Final synthesis step that combines all sub-insights."""
    prompt = "Synthesize a coherent final answer from the following insights:\n\n"
    prompt += "\n".join(f"- {ans}" for ans in sub_answers)
    prompt += "\n\nAnswer in a clear, concise paragraph."

    return llm_call(prompt)


@trace("Research Agent")
def research_agent(topic: str) -> str:
    """Main research agent entrypoint.

    Demonstrates nested steps, tool calls, LLM calls, and explicit
    phase management via the step() context manager.
    """
    tc = current_trace()
    if not tc:
        # Should not happen in normal usage of this example
        raise RuntimeError("No active trace context")

    print(f"   Researching: {topic}")

    # Phase 1: Broad search
    with tc.step("Broad Search", "phase"):
        initial_results = web_search(f"latest research on {topic}")

    # Phase 2: Targeted sub-investigations
    subtopics = ["technical foundations", "real-world deployments", "open challenges"]
    sub_answers: list[str] = []

    for subtopic in subtopics:
        with tc.step(f"Investigate: {subtopic}", "subtask"):
            print(f"     → {subtopic}")
            results = web_search(f"{subtopic} {topic}")
            insight = analyze_results(subtopic, results)
            sub_answers.append(insight)

    # Phase 3: Final synthesis
    with tc.step("Final Synthesis", "phase"):
        final_answer = synthesize_answer(sub_answers)

    print("   Research complete.\n")
    return final_answer


if __name__ == "__main__":
    topic = "observability for LLM agents"

    print("═" * 60)
    print("FlowTrace Example — Mock Research Agent")
    print("═" * 60)
    print(f"Topic: {topic}\n")

    with TraceContext(name=f"Research: {topic}") as tc:
        answer = research_agent(topic)

    print("Final Answer:")
    print("─" * 40)
    print(answer.strip())
    print("─" * 40)

    # Persist trace so it appears in the dashboard and CLI tools
    trace_model = Trace.from_tracer_context(tc)
    store = SQLiteStore()
    trace_id = store.save_trace(trace_model)

    print(f"\nTrace saved to SQLite with ID: {trace_id}")
    print("\nNext steps:")
    print("   flowtrace serve          # start the Swiss-minimal dashboard")
    print("   # then open http://127.0.0.1:8080")
    print("\nOr explore via CLI:")
    print("   flowtrace list")
    print("   flowtrace view " + trace_id)
    print("\nThe graph will show nested phases, LLM calls (with raw prompts),")
    print("tool invocations, durations, and token counts.")
    print("═" * 60)
