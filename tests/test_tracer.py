"""Tests for the FlowTrace core tracer.

Covers context manager, decorator, step recording, LLM and tool
instrumentation, timing, error handling and uniqueness guarantees.
"""

import time
import uuid
from typing import Any

import pytest

from flowtrace.tracer import (
    TraceContext,
    current_trace,
    trace,
)


def test_trace_context_records_nested_steps():
    """Verify that TraceContext builds a correct nested step tree."""
    with TraceContext(name="integration-test") as tc:
        tc.record_step("Initialize", "setup", {"version": "0.1.0"})

        with tc.step("Main Phase", "phase", {"mode": "research"}) as phase_step:
            tc.record_step("Subtask A", "action")
            tc.record_llm_call(
                prompt="Summarize the following: ...",
                response="Summary here.",
                tokens=87,
                latency=0.42,
            )

        tc.record_tool_call(
            name="wikipedia",
            args={"query": "LLM agents"},
            result={"summary": "long text..."},
        )

    root = tc.get_root_step()

    assert root.name == "integration-test"
    assert root.kind == "trace"
    assert root.duration > 0.0
    assert len(root.children) == 3  # init, main-phase, tool

    # Check nested structure
    phase = root.children[1]
    assert phase.name == "Main Phase"
    assert phase.kind == "phase"
    assert len(phase.children) == 2  # subtask + llm
    assert phase.children[0].name == "Subtask A"
    assert phase.children[1].kind == "llm"

    # Check content propagation
    llm_step = phase.children[1]
    assert llm_step.content["tokens"] == 87
    assert llm_step.content["prompt"].startswith("Summarize")

    tool_step = root.children[2]
    assert tool_step.kind == "tool"
    assert tool_step.content["name"] == "wikipedia"


def test_decorator_wraps_llm_and_tool_calls():
    """The @trace decorator should wrap functions and integrate with active context."""

    @trace("ResearchAgent")
    def research_agent(query: str) -> str:
        tc = current_trace()
        assert tc is not None
        tc.record_llm_call("Tell me about " + query, "Agentic systems are...", 45)
        tc.record_tool_call("search", {"q": query}, "result data")
        return "final answer"

    with TraceContext(name="RootTrace") as root_context:
        result = research_agent("observability")

    assert result == "final answer"

    root = root_context.get_root_step()
    assert root.name == "RootTrace"
    assert len(root.children) == 1

    agent_step = root.children[0]
    assert agent_step.name == "ResearchAgent"
    assert len(agent_step.children) == 2

    llm = next(s for s in agent_step.children if s.kind == "llm")
    tool = next(s for s in agent_step.children if s.kind == "tool")

    assert llm.content["tokens"] == 45
    assert tool.content["name"] == "search"


def test_trace_id_generation_is_unique():
    """Every TraceContext must receive a unique UUID."""
    trace_ids = set()

    for _ in range(25):
        with TraceContext() as tc:
            trace_ids.add(tc.trace_id)

    assert len(trace_ids) == 25
    # Verify they are valid uuids
    for tid in trace_ids:
        assert len(tid) == 36
        uuid.UUID(tid)


def test_step_timing_is_accurate():
    """Recorded durations should reflect actual elapsed time."""
    with TraceContext() as tc:
        with tc.step("FastStep"):
            pass

        with tc.step("SlowStep"):
            time.sleep(0.12)

    root = tc.get_root_step()
    slow = next((s for s in root.children if s.name == "SlowStep"), None)
    assert slow is not None
    assert slow.duration >= 0.11


def test_error_handling_in_context():
    """Exceptions are recorded in the trace tree."""
    try:
        with TraceContext(name="error-demo") as tc:
            tc.record_step("Before")
            with tc.step("RiskyOperation"):
                raise RuntimeError("Simulated failure")
    except RuntimeError:
        pass  # expected

    root = tc.get_root_step()
    assert any(
        "failed" in c.content.get("status", "") or "error" in str(c.content).lower()
        for c in root.children
    )


def test_current_trace_outside_context_is_none():
    """current_trace() returns None when no context is active."""
    assert current_trace() is None


def test_trace_without_active_context_creates_new():
    """Decorator should create a top-level trace when used outside context."""

    @trace(name="Standalone")
    def standalone():
        return current_trace().trace_id

    trace_id = standalone()

    # The test cannot easily capture the created context because it is
    # cleaned up on exit. We simply assert that a trace_id was produced.
    assert len(trace_id) == 36
