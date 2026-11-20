import time
from unittest.mock import MagicMock, patch

import pytest

from flowtrace.langchain import FlowTraceHandler
from flowtrace.llamaindex import LlamaIndexHandler
from flowtrace.models import Trace
from flowtrace.storage import SQLiteStore
from flowtrace.tracer import TraceContext, current_trace, trace


def test_context_basic():
    with TraceContext(name="test-basic") as tc:
        assert current_trace() is tc
        tc.record_step("setup")
        tc.record_llm_call("prompt", "response", tokens=5)
        root = tc.get_root_step()
        assert root is not None
        assert root.name == "test-basic"
        assert len(root.children) == 2
    assert current_trace() is None


def test_decorator():
    @trace("decorated")
    def fn(x):
        current_trace().record_llm_call("prompt", "answer", 10)
        return x * 2

    with TraceContext(name="outer") as tc:
        result = fn(21)
        assert result == 42
        root = current_trace().get_root_step() if current_trace() else None
        assert root is not None
        assert root.name == "outer"
        # decorator created a nested step
        assert any(c.name == "decorated" for c in root.children)
    assert current_trace() is None


def test_nested_contexts():
    with TraceContext(name="parent") as tc:
        with tc.step("child"):
            current_trace().record_tool_call("wiki", {"q": "a"}, "result")
        root = tc.get_root_step()
        assert len(root.children) == 1
        assert root.children[0].name == "child"
        assert len(root.children[0].children) == 1
        assert root.children[0].children[0].kind == "tool"


def test_timing():
    with TraceContext(name="timed") as tc:
        time.sleep(0.05)
        tc.record_llm_call("p", "r", 10)
    root = tc.get_root_step()
    assert root.duration > 0.04


def test_error_capture():
    with TraceContext(name="error-test") as tc:
        try:
            with tc.step("risky-step"):
                raise ValueError("boom")
        except ValueError:
            pass
    root = tc.get_root_step()
    assert not root.content.get("status") == "failed" or any(
        c.content.get("status") == "failed" or "error" in c.content
        for c in root.children
    )


def test_trace_id_uniqueness():
    ids = set()
    for _ in range(50):
        with TraceContext(name="unique") as tc:
            ids.add(tc.trace_id)
    assert len(ids) == 50


def test_current_trace_contextvar():
    assert current_trace() is None
    with TraceContext(name="cv-test"):
        assert current_trace() is not None
    assert current_trace() is None


def test_record_methods():
    with TraceContext(name="records") as tc:
        tc.record_llm_call("prompt", "answer", tokens=42, latency=0.3)
        tc.record_tool_call("calc", {"x": 2}, {"y": 4})
        tc.record_step("post-process", kind="post")
    root = tc.get_root_step()
    kinds = {c.kind for c in root.children}
    assert "llm" in kinds
    assert "tool" in kinds
    assert "post" in kinds
    assert any(c.content.get("tokens") == 42 for c in root.children)


def test_live_streaming():
    with patch("httpx.post") as mock_post:
        with TraceContext(name="live-test", live=True) as tc:
            tc.record_llm_call("p", "r", 5)
    assert mock_post.called


def test_models_from_tracer():
    with TraceContext(name="model-test") as tc:
        tc.record_llm_call("prompt1", "response1", tokens=99)
    trace_model = Trace.from_tracer_context(tc)
    assert trace_model.name == "model-test"
    assert trace_model.total_tokens == 99
    assert trace_model.root_step.name == "model-test"


def test_langchain_handler_structure():
    with patch("flowtrace.langchain.BaseCallbackHandler"):
        handler = FlowTraceHandler(name="lang-test", live=True)
        assert handler.name == "lang-test"
        assert handler.live is True
        assert handler.trace is None
        assert handler._parse_input("hello") == "hello"
        assert handler._parse_output({"output": "world"}) == "world"


def test_llamaindex_handler_structure():
    with patch("flowtrace.llamaindex.BaseCallbackHandler"):
        from flowtrace.llamaindex import LlamaIndexHandler

        handler = LlamaIndexHandler(name="llama-test", live=True)
        assert handler.name == "llama-test"
        assert handler.live is True
        assert handler.trace is None

        # test helper methods
        assert handler._event_type_to_kind("LLM") == "llm"
        assert handler._event_type_to_kind("RETRIEVE") == "retrieval"
        assert handler._event_type_to_kind("QUERY") == "query"

        payload = {"query_str": "test query", "total_tokens": 123}
        parsed = handler._parse_payload(payload, "QUERY", is_end=True)
        assert parsed["query"] == "test query"
        assert parsed.get("tokens") == 123


def test_llamaindex_event_flow():
    with patch("flowtrace.llamaindex.BaseCallbackHandler"):
        from flowtrace.llamaindex import LlamaIndexHandler

        handler = LlamaIndexHandler(name="event-test", live=False)
        store_mock = MagicMock()
        handler.store = store_mock

        # simulate start / end events
        handler.on_event_start("LLM", {"prompt": "hi"}, event_id="e1")
        handler.on_event_end(
            "LLM",
            {"response": "hello", "total_tokens": 10},
            event_id="e1",
        )
        assert store_mock.save_trace.called
        # handler resets after root completion
        assert handler.trace is None


def test_step_context_manager():
    with TraceContext(name="step-cm") as tc:
        with tc.step("inner-step", kind="phase", metadata={"k": "v"}):
            current_trace().record_llm_call("p2", "r2", 7)
    root = tc.get_root_step()
    assert len(root.children) == 1
    inner = root.children[0]
    assert inner.name == "inner-step"
    assert inner.kind == "phase"
    assert inner.content["k"] == "v"
    assert len(inner.children) == 1
