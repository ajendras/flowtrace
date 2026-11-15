"""Tests for the FlowTrace tracer.

Verifies TraceContext, @trace decorator, step nesting, error capture,
timing, current_trace contextvar, ID uniqueness, and LangChain handler.
"""

from __future__ import annotations

import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from flowtrace.tracer import TraceContext, trace, current_trace
from flowtrace.langchain import FlowTraceHandler
from flowtrace.models import Trace


def test_trace_context_creates_root():
    with TraceContext("test-root") as tc:
        assert tc.name == "test-root"
        assert tc.root_step.name == "test-root"
        assert tc.root_step.kind == "trace"
        assert len(tc.root_step.children) == 0
    assert tc.root_step.duration > 0
    assert current_trace() is None


def test_nested_step_structure():
    with TraceContext("nested") as tc:
        tc.record_step("setup", "setup")
        with tc.step("main-phase", "phase") as step:
            tc.record_llm_call("prompt", "response", 50, 0.2)
            with tc.step("subtool", "tool"):
                pass
    root = tc.root_step
    assert len(root.children) == 2
    assert root.children[0].kind == "setup"
    phase = root.children[1]
    assert phase.kind == "phase"
    assert len(phase.children) == 2
    assert phase.children[0].kind == "llm"
    assert phase.children[1].kind == "tool"


def test_decorator_top_level():
    trace_container = []

    @trace("top-level", kind="llm")
    def top_agent():
        tc = current_trace()
        trace_container.append(tc)
        tc.record_llm_call("system prompt", "final answer", 42)
        return "success"

    result = top_agent()
    assert result == "success"
    assert len(trace_container) == 1
    tc = trace_container[0]
    root = tc.root_step
    assert root.name == "top-level"
    assert root.kind == "trace"
    assert len(root.children) == 1
    decorated = root.children[0]
    assert decorated.name == "top-level"
    assert decorated.kind == "llm"
    assert any(c.kind == "llm" for c in decorated.children)


def test_decorator_nested():
    with TraceContext("parent-trace") as tc:
        @trace("nested-step")
        def nested():
            current_trace().record_tool_call("calc", {"a": 1}, 2)
            return 42

        nested()
    root = tc.root_step
    assert len(root.children) == 1
    nested_step = root.children[0]
    assert nested_step.name == "nested-step"
    assert nested_step.kind == "step"
    assert len(nested_step.children) == 1
    tool = nested_step.children[0]
    assert tool.kind == "tool"


def test_llm_and_tool_recording():
    with TraceContext("recorder") as tc:
        tc.record_llm_call("What is AI?", "AI is...", 100, 0.35)
        tc.record_tool_call("web_search", {"query": "flowtrace"}, {"results": [...]})
        tc.record_step("custom", "phase", {"note": "done"})
    root = tc.root_step
    assert len(root.children) == 3
    kinds = [c.kind for c in root.children]
    assert "llm" in kinds
    assert "tool" in kinds
    assert "phase" in kinds
    llm = next(c for c in root.children if c.kind == "llm")
    assert llm.content["tokens"] == 100
    assert llm.content["prompt"] == "What is AI?"


def test_timing_is_accurate():
    with TraceContext("timer") as tc:
        with tc.step("quick"):
            pass
        with tc.step("slower"):
            time.sleep(0.015)
    root = tc.root_step
    quick = root.children[0]
    slower = root.children[1]
    assert quick.duration < 0.005
    assert slower.duration > 0.01
    assert root.duration > slower.duration


def test_error_capture():
    with pytest.raises(ValueError, match="boom"):
        with TraceContext("error-test") as tc:
            with tc.step("risky"):
                raise ValueError("boom")
    root = tc.root_step
    assert len(root.children) == 1
    risky = root.children[0]
    assert risky.name == "risky"
    assert risky.kind == "step"
    assert risky.content.get("status") == "failed"
    assert risky.content.get("error") == "boom"
    assert root.content.get("status") == "failed"


def test_current_trace_contextvar():
    assert current_trace() is None
    with TraceContext("outer") as outer:
        assert current_trace() is outer
        with TraceContext("inner") as inner:
            assert current_trace() is inner
        assert current_trace() is outer  # restored by reset
    assert current_trace() is None


def test_unique_ids():
    ids = set()
    for _ in range(50):
        tc = TraceContext("unique")
        ids.add(tc.trace_id)
        with tc.step("s1") as s1:
            ids.add(s1.id)
            with tc.step("s2") as s2:
                ids.add(s2.id)
    assert len(ids) == 50 * 3


@patch("flowtrace.langchain.SQLiteStore")
def test_langchain_handler_records_steps(mock_sqlite_class):
    mock_store = MagicMock()
    mock_sqlite_class.return_value = mock_store
    handler = FlowTraceHandler(name="TestLangchainAgent")
    run_id = uuid.uuid4()
    handler.on_llm_start(
        serialized={"name": "gpt-test"},
        prompts=["Tell me a joke"],
        run_id=run_id,
    )
    assert handler.trace is not None
    assert handler.root_run_id == str(run_id)
    resp = MagicMock()
    resp.llm_output = {"token_usage": {"total_tokens": 35}}
    handler.on_llm_end(resp, run_id=run_id)
    mock_store.save_trace.assert_called_once()
    saved = mock_store.save_trace.call_args[0][0]
    assert isinstance(saved, Trace)
    assert saved.name == "TestLangchainAgent"
    assert saved.total_tokens == 35
    llm_steps = [s for s in saved.root_step.children if s.kind == "llm"]
    assert len(llm_steps) == 1
    assert llm_steps[0].content["status"] == "completed"
