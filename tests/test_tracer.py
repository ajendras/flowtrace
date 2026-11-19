import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from flowtrace.tracer import TraceContext, current_trace, trace, Step
from flowtrace.models import Trace, Step as ModelStep


def test_trace_context_creates_root():
    with TraceContext(name="test-trace") as tc:
        assert tc.name == "test-trace"
        assert tc.root_step.name == "test-trace"
        assert tc.root_step.kind == "trace"
        assert len(tc.root_step.children) == 0
    assert tc.root_step.duration > 0
    assert tc.success is True


def test_nested_steps():
    with TraceContext(name="parent") as tc:
        with tc.step("child1", "phase"):
            tc.record_step("grandchild", "llm")
        tc.record_step("sibling", "tool")
    assert len(tc.root_step.children) == 2
    assert tc.root_step.children[0].name == "child1"
    assert len(tc.root_step.children[0].children) == 1
    assert tc.root_step.children[0].children[0].kind == "llm"
    assert tc.root_step.children[1].kind == "tool"


def test_decorator():
    @trace("decorated")
    def fn(x):
        current_trace().record_llm_call("prompt", "answer", 10)
        return x * 2

    with TraceContext(name="outer"):
        result = fn(21)
    assert result == 42
    root = current_trace().get_root_step() if current_trace() else None
    assert root is not None
    assert any(s.name == "decorated" for s in root.children)


def test_record_llm_and_tool():
    with TraceContext(name="records") as tc:
        tc.record_llm_call("hello", "world", 42, latency=0.1)
        tc.record_tool_call("search", {"q": "foo"}, "result")
    steps = tc.root_step.children
    assert len(steps) == 2
    assert steps[0].kind == "llm"
    assert steps[0].content["tokens"] == 42
    assert steps[0].content["latency_ms"] == 100.0
    assert steps[1].kind == "tool"
    assert steps[1].content["arguments"]["q"] == "foo"


def test_error_capture():
    with pytest.raises(ValueError):
        with TraceContext(name="error-test") as tc:
            with tc.step("risky"):
                raise ValueError("boom")
    assert tc.success is False
    assert any(s.content.get("status") == "failed" for s in tc.root_step.children)


def test_current_trace_contextvar():
    assert current_trace() is None
    with TraceContext(name="ctx") as tc:
        assert current_trace() is tc
    assert current_trace() is None


def test_unique_ids():
    ids = set()
    for _ in range(50):
        with TraceContext(name="unique") as tc:
            ids.add(tc.trace_id)
    assert len(ids) == 50


def test_step_to_dict():
    s = Step(id="1", name="test", kind="llm", duration=0.123)
    d = s.to_dict()
    assert d["duration_ms"] == 123.0
    assert d["kind"] == "llm"


def test_model_from_tracer():
    tc = TraceContext(name="model-test")
    with tc.step("llm-step", "llm", {"tokens": 50}):
        pass
    model = Trace.from_tracer_context(tc)
    assert model.name == "model-test"
    assert model.total_tokens == 50
    assert isinstance(model.root_step, ModelStep)


@patch("flowtrace.langchain.BaseCallbackHandler", new=object)
def test_langchain_handler_import_guard():
    from flowtrace import FlowTraceHandler

    with pytest.raises(ImportError):
        FlowTraceHandler()


def test_langchain_handler_structure():
    # Mock the real base class so the handler can be instantiated
    with patch("flowtrace.langchain.BaseCallbackHandler"):
        from flowtrace.langchain import FlowTraceHandler

        handler = FlowTraceHandler(name="lang-test", live=False)
        assert handler.name == "lang-test"
        assert handler.trace is None
        assert len(handler.steps) == 0


@patch("flowtrace.llamaindex.BaseCallbackHandler", new=object)
def test_llamaindex_handler_import_guard():
    from flowtrace import LlamaIndexHandler

    with pytest.raises(ImportError):
        LlamaIndexHandler()


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
        assert parsed["tokens"] == 123


def test_llamaindex_event_flow():
    with patch("flowtrace.llamaindex.BaseCallbackHandler"):
        from flowtrace.llamaindex import LlamaIndexHandler
        from flowtrace.storage import SQLiteStore

        handler = LlamaIndexHandler(name="event-test", live=False)
        store_mock = MagicMock()
        handler.store = store_mock

        # simulate start / end events
        handler.on_event_start("LLM", {"prompt": "hi"}, event_id="e1")
        assert len(handler.steps) == 1
        assert handler.steps["e1"].kind == "llm"

        handler.on_event_end("LLM", {"response": "hello", "total_tokens": 5}, event_id="e1")
        assert handler.steps["e1"].content["response"] == "hello"
        assert handler.steps["e1"].duration > 0

        # root finalization would be triggered on root event
        handler.root_event_id = "e1"
        handler.trace = MagicMock()
        handler._maybe_finalize("e1")
        store_mock.save_trace.assert_called_once()
