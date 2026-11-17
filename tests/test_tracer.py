"""Test suite for FlowTrace.

Covers TraceContext, @trace decorator, recording methods, timing, errors,
current_trace contextvar semantics, UUID uniqueness, model conversion,
and the optional LangChain FlowTraceHandler (skipped gracefully when
langchain-core is not installed).
"""

import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from flowtrace.models import Trace
from flowtrace.tracer import TraceContext, current_trace, trace


def test_trace_context_initialization():
    with TraceContext(name="TestWorkflow") as tc:
        assert tc.name == "TestWorkflow"
        assert uuid.UUID(tc.trace_id)
        assert tc.root_step.name == "TestWorkflow"
        assert tc.root_step.kind == "trace"
        assert tc.root_step.start > 0
        assert tc.success is True
        assert len(tc.root_step.children) == 0


def test_nested_step_context_manager():
    with TraceContext(name="Parent") as tc:
        with tc.step("Setup", "setup") as s1:
            assert s1.name == "Setup"
            assert s1.kind == "setup"
            with tc.step("Core", "phase") as s2:
                tc.record_step("Substep")
                assert len(s2.children) == 1
        assert len(tc.root_step.children) == 1
        parent = tc.root_step.children[0]
        assert parent.name == "Setup"
        assert len(parent.children) == 1
        assert parent.children[0].name == "Core"
        assert len(parent.children[0].children) == 1


def test_decorator_as_top_level_trace():
    @trace("MyAgent")
    def run_agent(query):
        current = current_trace()
        assert current is not None
        current.record_llm_call("prompt here", "answer", 25)
        return "result"

    result = run_agent("test query")
    assert result == "result"


def test_decorator_inside_existing_context():
    with TraceContext("MainTrace") as outer:
        @trace("NestedFunc")
        def nested():
            current_trace().record_tool_call("db_query", {"id": 1}, "data")  # type: ignore[attr-defined]
            return 99

        nested()
        assert len(outer.root_step.children) == 1
        nested_step = outer.root_step.children[0]
        assert nested_step.name == "NestedFunc"
        assert nested_step.kind == "step"
        assert len(nested_step.children) == 1
        assert nested_step.children[0].kind == "tool"


def test_llm_and_tool_recording():
    with TraceContext() as tc:
        tc.record_llm_call(
            prompt="Classify this text",
            response="positive",
            tokens=42,
            latency=0.23,
        )
        tc.record_tool_call(
            name="web_search",
            args={"query": "flowtrace"},
            result={"status": "success", "results": 5},
        )
    steps = tc.root_step.children
    assert len(steps) == 2
    llm_step = steps[0]
    assert llm_step.kind == "llm"
    assert llm_step.content["prompt"] == "Classify this text"
    assert llm_step.content["tokens"] == 42
    assert llm_step.content.get("latency_ms") == 230.0
    tool_step = steps[1]
    assert tool_step.kind == "tool"
    assert tool_step.name == "web_search"
    assert tool_step.content["arguments"]["query"] == "flowtrace"


def test_timing_and_duration():
    with TraceContext() as tc:
        with tc.step("TimedStep"):
            time.sleep(0.05)
    assert tc.root_step.duration > 0.04
    assert tc.root_step.children[0].duration > 0.04


def test_error_handling_in_context():
    with pytest.raises(RuntimeError, match="boom"):
        with TraceContext(name="FailingTrace") as tc:
            tc.record_step("GoodStep")
            with tc.step("FailingStep"):
                raise RuntimeError("boom")
    assert tc.success is False
    error_step = tc.root_step.children[1]
    assert error_step.content.get("status") == "failed"
    assert "boom" in str(error_step.content.get("error"))


def test_current_trace_contextvar_isolation():
    assert current_trace() is None
    with TraceContext(name="Outer") as outer:
        assert current_trace() is outer
        with TraceContext(name="Inner") as inner:
            assert current_trace() is inner
        assert current_trace() is outer
    assert current_trace() is None


def test_trace_id_uniqueness():
    trace_ids = {TraceContext().trace_id for _ in range(20)}
    assert len(trace_ids) == 20


def test_model_conversion():
    with TraceContext(name="ConversionTest") as tc:
        tc.record_llm_call("hello", "world", 10)
        tc.record_step("final", "phase")
    trace_model = Trace.from_tracer_context(tc)
    assert trace_model.id == tc.trace_id
    assert trace_model.name == "ConversionTest"
    assert trace_model.total_tokens == 10
    assert trace_model.success is True
    assert len(trace_model.root_step.children) == 2


def test_live_streaming_mocked():
    with patch("flowtrace.tracer.httpx") as mock_httpx:
        with TraceContext(name="LiveTest", live=True) as tc:
            tc.record_step("live-step")
            assert mock_httpx.Client.call_count > 0


def test_flowtrace_handler():
    pytest.importorskip("langchain_core", reason="langchain-core not installed")
    from flowtrace.langchain import FlowTraceHandler
    from langchain_core.callbacks.base import BaseCallbackHandler

    assert issubclass(FlowTraceHandler, BaseCallbackHandler)

    handler = FlowTraceHandler(name="TestAgent", live=False)
    assert handler.name == "TestAgent"
    assert not handler.live

    # Mock internals to test event flow without real DB or WS
    mock_trace = MagicMock()
    mock_trace.root_step = MagicMock(children=[])
    handler.trace = mock_trace
    handler.steps = {}
    handler.store = MagicMock()
    handler.root_run_id = None

    run_id = uuid.uuid4()
    handler.on_llm_start(
        serialized={"name": "gpt-4o"},
        prompts=["Hello world"],
        run_id=run_id,
    )
    assert str(run_id) in handler.steps
    step = handler.steps[str(run_id)]
    assert step.kind == "llm"
    assert "Hello world" in step.content["prompt"]

    response = MagicMock()
    response.llm_output = {"token_usage": {"total_tokens": 15}}
    handler.on_llm_end(response=response, run_id=run_id)
    assert step.content["status"] == "completed"
    assert step.content.get("tokens") == 15

    # Simulate root completion
    handler.root_run_id = str(run_id)
    handler._maybe_finalize(str(run_id))
    assert handler.store.save_trace.call_count >= 1
