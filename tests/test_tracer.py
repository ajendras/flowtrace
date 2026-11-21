"""Tests for the FlowTrace core tracer, handlers and models."""

from __future__ import annotations

import time
import uuid
from unittest.mock import patch

import pytest

from flowtrace.models import Step as ModelStep
from flowtrace.models import Trace
from flowtrace.storage import SQLiteStore
from flowtrace.tracer import TraceContext, current_trace, trace


def test_trace_context_basic():
    with TraceContext(name="TestBasic", live=False) as tc:
        tc.record_step("Setup", kind="setup")
        tc.record_llm_call("prompt", "response", tokens=42, latency=0.05)
    assert len(tc.root_step.children) == 2
    setup = tc.root_step.children[0]
    assert setup.name == "Setup"
    assert setup.kind == "setup"
    assert setup.duration == 0.0
    llm = tc.root_step.children[1]
    assert llm.kind == "llm"
    assert llm.duration == 0.05
    assert tc.root_step.duration > 0.0
    trace_model = Trace.from_tracer_context(tc)
    assert trace_model.total_tokens == 42
    assert trace_model.duration > 0.0
    assert trace_model.success


def test_nested_steps_and_timing():
    with TraceContext(name="TestNested", live=False) as tc:
        with tc.step("Parent", kind="chain"):
            time.sleep(0.01)
            with tc.step("ChildLLM", kind="llm"):
                time.sleep(0.005)
                tc.record_llm_call(
                    "What is AI?", "AI is...", tokens=25, latency=0.003
                )
            tc.record_tool_call("web_search", {"q": "ai"}, {"status": "ok"}, latency=0.002)
    assert len(tc.root_step.children) == 1
    parent = tc.root_step.children[0]
    assert parent.name == "Parent"
    assert len(parent.children) == 2
    llm_step = next(s for s in parent.children if s.kind == "llm")
    assert len(llm_step.children) == 1
    assert llm_step.duration > 0.0
    tool_step = next(s for s in parent.children if s.kind == "tool")
    assert tool_step.duration > 0.0
    assert parent.duration > 0.015
    assert tc.root_step.duration > 0.015


def test_decorator():
    @trace("DecoratedAgent")
    def agent_task(x):
        current_trace().record_tool_call("calc", {"x": x}, {"y": x * 2})
        return x * 2

    agent_task(5)
    assert True  # runs without error


def test_error_capture():
    with pytest.raises(ValueError):
        with TraceContext(name="ErrorTest") as tc:
            tc.record_step("FailingStep")
            raise ValueError("test error")
    assert not tc.success
    assert any(
        "error" in (c.content if hasattr(c, "content") else {})
        for c in tc.root_step.children
    ) or "error" in tc.root_step.content


def test_current_trace():
    with TraceContext(name="CurrentTest") as tc:
        assert current_trace() is tc
        current_trace().record_step("Inside")
    assert len(tc.root_step.children) == 1


def test_unique_ids():
    ids = set()
    with TraceContext("IDTest") as tc:
        for i in range(5):
            with tc.step(f"step{i}"):
                ids.add(tc.root_step.children[-1].id)
    assert len(ids) == 5


@pytest.mark.parametrize("live", [False, True])
def test_live_update(live: bool):
    with patch("flowtrace.tracer.httpx.post") as mock_post:
        with TraceContext(name="LiveTest", live=live) as _tc:
            pass
        if live:
            assert mock_post.called
        else:
            assert not mock_post.called


def test_langchain_handler_export():
    pytest.importorskip("langchain_core")
    from flowtrace.langchain import FlowTraceHandler

    store = SQLiteStore()
    handler = FlowTraceHandler(name="TestAgent", live=False)
    handler.store = store
    chain_id = uuid.uuid4()
    llm_id = uuid.uuid4()
    handler.on_chain_start(
        serialized={"name": "TestAgent"},
        inputs="user query",
        run_id=chain_id,
    )
    handler.on_llm_start(
        serialized={"name": "gpt-4o"},
        prompts=["test prompt"],
        run_id=llm_id,
        parent_run_id=chain_id,
    )
    mock_response = type(
        "MockResponse",
        (),
        {
            "content": "test response",
            "llm_output": {"token_usage": {"total_tokens": 35}},
        },
    )()
    handler.on_llm_end(mock_response, run_id=llm_id)
    handler.on_chain_end({"output": "final"}, run_id=chain_id)
    traces = store.list_traces(limit=10)
    trace_obj = None
    for summary in traces:
        if summary.name == "TestAgent":
            trace_obj = store.get_trace(summary.id)
            break
    assert trace_obj is not None
    assert trace_obj.total_tokens == 35
    assert trace_obj.success


def test_llamaindex_handler_structure():
    pytest.importorskip("llama_index.core")
    from flowtrace.llamaindex import LlamaIndexHandler

    store = SQLiteStore()
    handler = LlamaIndexHandler(name="TestIndex", live=False)
    handler.store = store
    query_id = str(uuid.uuid4())
    llm_id = str(uuid.uuid4())
    handler.on_event_start(
        event_type=type("ET", (), {"value": "QUERY"})(),
        payload={"query_str": "what is flowtrace?"},
        event_id=query_id,
    )
    handler.on_event_start(
        event_type=type("ET", (), {"value": "LLM"})(),
        payload={"prompt": "test prompt"},
        event_id=llm_id,
        parent_id=query_id,
    )
    handler.on_event_end(
        event_type=type("ET", (), {"value": "LLM"})(),
        payload={"response": "It visualizes agent workflows", "total_tokens": 15},
        event_id=llm_id,
    )
    handler.on_event_end(
        event_type=type("ET", (), {"value": "QUERY"})(),
        payload={"response": "final answer"},
        event_id=query_id,
    )
    traces = store.list_traces(limit=10)
    trace_obj = None
    for summary in traces:
        if summary.name == "TestIndex":
            trace_obj = store.get_trace(summary.id)
            break
    assert trace_obj is not None
    assert trace_obj.total_tokens > 0
    assert trace_obj.success


def test_model_conversion():
    with TraceContext("ModelTest") as tc:
        tc.record_llm_call("p", "r", 10)
    model = Trace.from_tracer_context(tc)
    assert isinstance(model.root_step, ModelStep)
    assert model.root_step.duration_ms > 0
    assert any(c.kind == "llm" for c in model.root_step.children)


def test_step_kinds():
    with TraceContext("KindTest") as tc:
        with tc.step("LLM", kind="llm"):
            pass
        with tc.step("Tool", kind="tool"):
            pass
        tc.record_step("Setup", kind="setup")
    kinds = [s.kind for s in tc.root_step.children]
    assert "llm" in kinds
    assert "tool" in kinds
    assert "setup" in kinds
