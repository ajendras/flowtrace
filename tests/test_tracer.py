"""Tests for the core tracer.

Covers TraceContext, decorator, nesting, error handling, current_trace,
dataclass ordering (the CI failure root cause), and both framework handlers.
"""

import pytest
from unittest.mock import MagicMock, patch

from flowtrace.tracer import TraceContext, current_trace, trace, Step
from flowtrace.langchain import FlowTraceHandler
from flowtrace.llamaindex import LlamaIndexHandler


def test_step_dataclass_order():
    """The root cause fix: non-default 'name' precedes fields with defaults."""
    s = Step("test-name", kind="test", content={"a": 1})
    assert s.name == "test-name"
    assert s.kind == "test"
    assert s.content == {"a": 1}
    assert s.duration == 0.0
    assert len(s.children) == 0
    assert isinstance(s.id, str)


def test_trace_context():
    with TraceContext(name="test-context", live=False) as tc:
        assert current_trace() is tc
        tc.record_llm_call("What is 2+2?", "4", tokens=5, latency=0.2)
        tc.record_tool_call("calc", {"op": "add"}, {"result": 4})
        with tc.step("substep", kind="phase", metadata={"key": "value"}):
            tc.record_step("inner", kind="step")
    assert tc.duration > 0
    assert tc.success is True
    assert tc.name == "test-context"
    assert len(tc.root_step.children) == 3  # llm, tool, substep
    assert tc.root_step.children[0].kind == "llm"
    assert tc.root_step.children[0].content["tokens"] == 5
    assert tc.root_step.children[2].name == "substep"
    assert len(tc.root_step.children[2].children) == 1


def test_error_in_context():
    with pytest.raises(RuntimeError, match="boom"):
        with TraceContext("error") as tc:
            with tc.step("failing-step"):
                raise RuntimeError("boom")
    assert tc.success is False
    assert tc.root_step.children[0].content.get("status") == "failed"
    assert "boom" in tc.root_step.children[0].content.get("error", "")


def test_decorator():
    @trace("decorated-fn")
    def sample_fn(x):
        tc = current_trace()
        tc.record_llm_call("prompt", "response", 3)
        return x * 2

    result = sample_fn(21)
    assert result == 42


def test_unique_trace_ids():
    ids = {TraceContext().id for _ in range(50)}
    assert len(ids) == 50


def test_step_nesting_with_stack():
    with TraceContext("nest-test") as tc:
        with tc.step("level1"):
            with tc.step("level2"):
                current_trace().record_step("level3")
    root = tc.root_step
    assert root.children[0].name == "level1"
    assert root.children[0].children[0].name == "level2"
    assert root.children[0].children[0].children[0].name == "level3"


@patch("flowtrace.langchain.SQLiteStore")
def test_langchain_handler(mock_store):
    mock_store.return_value = MagicMock()
    handler = FlowTraceHandler(name="test-agent", live=False)
    run_id = "123e4567-e89b-12d3-a456-426614174000"
    handler.on_chain_start(
        {"name": "TestChain"},
        {"input": "hello"},
        run_id=run_id,
        parent_run_id=None,
    )
    handler.on_llm_start(
        {"name": "TestLLM"},
        ["prompt here"],
        run_id="llm-run-id",
        parent_run_id=run_id,
    )
    mock_response = MagicMock()
    mock_response.llm_output = {"token_usage": {"total_tokens": 10}}
    mock_response.content = "response"
    handler.on_llm_end(mock_response, run_id="llm-run-id")
    handler.on_chain_end({"output": "final"}, run_id=run_id)
    assert mock_store.return_value.save_trace.called
    saved = mock_store.return_value.save_trace.call_args[0][0]
    assert saved.name == "test-agent"


@patch("flowtrace.llamaindex.SQLiteStore")
def test_llamaindex_handler(mock_store):
    mock_store.return_value = MagicMock()
    handler = LlamaIndexHandler(name="test-index", live=False)
    event_id = "evt-1"
    handler.on_event_start("llm", {"query_str": "test query"}, event_id=event_id)
    handler.on_event_end(
        "llm",
        {"response": "test response", "total_tokens": 25},
        event_id=event_id,
    )
    assert mock_store.return_value.save_trace.called


def test_current_trace_outside_context():
    with pytest.raises(RuntimeError):
        current_trace()
