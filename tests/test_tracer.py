"""Tests for FlowTrace core tracer and LangChain handler.

Covers TraceContext, @trace decorator, step nesting, LLM/tool recording,
timing, error capture, contextvar semantics, ID uniqueness, and the
official LangChain callback handler.
"""

import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from flowtrace.tracer import TraceContext, current_trace, trace
from flowtrace.models import Trace


def test_trace_context_creates_root():
    with TraceContext(name="test-root") as tc:
        assert tc.name == "test-root"
        assert tc.trace_id is not None
        assert len(tc.trace_id) > 10
        assert current_trace() is tc
        assert tc.root_step.name == "test-root"
        assert tc.root_step.kind == "trace"
    assert tc.root_step.duration > 0.0


def test_nested_step_context_manager():
    with TraceContext("nested-test") as tc:
        with tc.step("parent", "phase"):
            tc.record_step("child1", "setup")
            with tc.step("child2", "llm") as child2:
                child2.content["tokens"] = 42
            tc.record_tool_call("search", {"q": "foo"}, "bar")
    root = tc.root_step
    assert len(root.children) == 1
    parent = root.children[0]
    assert parent.name == "parent"
    assert len(parent.children) == 3
    assert parent.children[0].name == "child1"
    assert parent.children[1].name == "child2"
    assert parent.children[1].content["tokens"] == 42
    assert parent.children[2].kind == "tool"
    assert parent.children[2].content["output"] == "bar"


def test_decorator_top_level():
    calls = []

    @trace("decorated-agent")
    def agent_fn(x):
        tc = current_trace()
        tc.record_llm_call("prompt", "answer", 10)
        calls.append(tc)
        return x * 2

    result = agent_fn(21)
    assert result == 42
    assert len(calls) == 1
    trace_obj = calls[0]
    assert trace_obj.name == "decorated-agent"
    root = trace_obj.root_step
    assert len(root.children) == 1
    assert root.children[0].kind == "llm"


def test_decorator_nested():
    with TraceContext("outer") as outer:
        @trace("inner-decorated")
        def inner():
            tc = current_trace()
            tc.record_step("nested-step")
            return "done"

        inner()
    assert len(outer.root_step.children) == 1
    inner_step = outer.root_step.children[0]
    assert inner_step.name == "inner-decorated"
    assert len(inner_step.children) == 1
    assert inner_step.children[0].name == "nested-step"


def test_record_llm_and_tool():
    with TraceContext("metrics") as tc:
        tc.record_llm_call(
            prompt="What is 2+2?",
            response="4",
            tokens=8,
            latency=0.15,
        )
        tc.record_tool_call("calc", {"op": "add"}, 4)
    root = tc.get_root_step()
    assert len(root.children) == 2
    llm = root.children[0]
    assert llm.kind == "llm"
    assert llm.content["tokens"] == 8
    assert llm.content["latency_ms"] == 150.0
    tool = root.children[1]
    assert tool.kind == "tool"
    assert tool.content["arguments"]["op"] == "add"


def test_error_capture():
    with pytest.raises(ValueError):
        with TraceContext("error-test") as tc:
            with tc.step("risky"):
                raise ValueError("boom")
    root = tc.root_step
    assert len(root.children) == 1
    err_step = root.children[0].children[0] if root.children[0].children else root.children[0]
    assert err_step.content.get("status") == "failed"
    assert "boom" in err_step.content.get("error", "")


def test_current_trace_contextvar_isolation():
    trace1 = None
    trace2 = None

    def task1():
        nonlocal trace1
        with TraceContext("task1"):
            trace1 = current_trace()

    def task2():
        nonlocal trace2
        with TraceContext("task2"):
            trace2 = current_trace()

    import threading
    t1 = threading.Thread(target=task1)
    t2 = threading.Thread(target=task2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert trace1 is not None
    assert trace2 is not None
    assert trace1.trace_id != trace2.trace_id


def test_uuid_uniqueness():
    ids = set()
    for _ in range(50):
        with TraceContext("uniq") as tc:
            ids.add(tc.trace_id)
            for i in range(5):
                with tc.step(f"step{i}"):
                    pass
    assert len(ids) == 50


def test_trace_from_tracer_context():
    with TraceContext("summary-test") as tc:
        tc.record_llm_call("hi", "hello", 5)
        tc.record_step("final", "step", {"status": "ok"})

    trace_model = Trace.from_tracer_context(tc)
    assert trace_model.name == "summary-test"
    assert trace_model.total_tokens == 5
    assert trace_model.success is True
    assert len(trace_model.root_step.children) == 2


@patch("flowtrace.langchain.SQLiteStore")
def test_langchain_handler_records_steps(mock_store_cls):
    pytest.importorskip("langchain_core")
    from flowtrace.langchain import FlowTraceHandler

    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store

    handler = FlowTraceHandler(name="TestAgent")

    # Top-level chain
    chain_id = uuid.uuid4()
    handler.on_chain_start(
        {"name": "TestChain"},
        {"query": "What time is it?"},
        run_id=chain_id,
        parent_run_id=None,
    )
    assert handler.trace is not None
    assert len(handler.trace.root_step.children) == 1
    chain_step = handler.trace.root_step.children[0]
    assert chain_step.name == "TestChain"
    assert chain_step.kind == "chain"
    assert chain_step.content["status"] == "started"

    # Nested LLM
    llm_id = uuid.uuid4()
    handler.on_llm_start(
        {"name": "ChatOpenAI"},
        ["System: You are helpful\nUser: hi"],
        run_id=llm_id,
        parent_run_id=chain_id,
    )
    assert len(chain_step.children) == 1
    llm_step = chain_step.children[0]
    assert llm_step.kind == "llm"
    assert "System: You are helpful" in llm_step.content["prompt"]

    # LLM end with tokens
    mock_response = MagicMock()
    mock_response.generations = [[MagicMock(text="Hello there!")]]
    mock_response.llm_output = {"token_usage": {"total_tokens": 23}}
    handler.on_llm_end(mock_response, run_id=llm_id)
    assert llm_step.content["response"] == "Hello there!"
    assert llm_step.content["tokens"] == 23
    assert llm_step.content["status"] == "completed"
    assert llm_step.duration > 0

    # Chain end triggers save + reset
    handler.on_chain_end({"answer": "2pm"}, run_id=chain_id)
    mock_store.save_trace.assert_called_once()
    assert handler.trace is None
    assert handler.root_run_id is None
    assert len(handler.steps) == 0
