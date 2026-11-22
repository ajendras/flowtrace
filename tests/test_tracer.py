"""Tests for the FlowTrace core tracer.

Covers TraceContext, decorator, live updates, error handling, timing,
uniqueness, and the official handlers for LangChain and LlamaIndex.
"""

from __future__ import annotations

import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from flowtrace.langchain import FlowTraceHandler
from flowtrace.llamaindex import LlamaIndexHandler
from flowtrace.tracer import TraceContext, current_trace, trace


@pytest.mark.parametrize("live", [False, True])
def test_live_update(live: bool):
    with patch("flowtrace.tracer.httpx.post") as mock_post:
        with TraceContext(name="live_test", live=live) as tc:
            tc.record_llm_call(
                prompt="What is observability?",
                response="It's seeing inside your agent",
                tokens=42,
            )
            with tc.step("tool step"):
                tc.record_tool_call("search", {"q": "flowtrace"}, "results")

        if live:
            assert mock_post.call_count > 0
            # last call contains final aggregated state
            _, kwargs = mock_post.call_args
            assert "/live" in kwargs["url"] if "url" in kwargs else "/live" in str(
                mock_post.call_args[0][0]
            )
            json_data = kwargs.get("json", {})
            assert json_data.get("name") == "live_test"
            assert json_data.get("success") is True
            assert json_data.get("total_tokens") == 42
        else:
            mock_post.assert_not_called()


def test_trace_context_basic():
    with TraceContext(name="basic") as tc:
        tc.record_step("setup")
        tc.record_llm_call("prompt", "response", 10)
    assert len(tc.root_step.children) == 2
    assert tc.root_step.children[0].name == "setup"
    assert tc.root_step.children[1].kind == "llm"
    assert tc.success
    assert tc.root_step.duration > 0


def test_nested_step_context():
    with TraceContext("nested") as tc:
        with tc.step("phase1", kind="phase", metadata={"model": "gpt-4"}):
            tc.record_tool_call("tool1", {"arg": "val"}, "output1")
            with tc.step("substep"):
                pass
    assert len(tc.root_step.children) == 1
    phase = tc.root_step.children[0]
    assert phase.name == "phase1"
    assert phase.kind == "phase"
    assert phase.content["model"] == "gpt-4"
    assert len(phase.children) == 2


def test_decorator():
    calls = []

    @trace("decorated_func")
    def sample_func(x: int) -> int:
        current_trace().record_step("computation")
        calls.append(x)
        return x * 2

    result = sample_func(21)
    assert result == 42
    assert len(calls) == 1


def test_current_trace_outside_raises():
    with pytest.raises(RuntimeError, match="No active trace"):
        current_trace()


def test_error_handling():
    with pytest.raises(ValueError, match="boom"):
        with TraceContext(name="failing") as tc:
            with tc.step("risky"):
                raise ValueError("boom")
    assert not tc.success
    step = tc.root_step.children[0]
    assert step.content.get("status") == "failed"
    assert "boom" in str(step.content.get("error"))


def test_timing():
    with TraceContext("timing") as tc:
        time.sleep(0.01)
        with tc.step("slow-step"):
            time.sleep(0.03)
    assert tc.root_step.duration >= 0.04
    assert tc.root_step.children[0].duration >= 0.03


def test_unique_trace_ids():
    ids = {TraceContext("test").trace_id for _ in range(5)}
    assert len(ids) == 5


def test_flowtrace_handler():
    with patch("flowtrace.langchain.SQLiteStore") as MockStore:
        mock_store = MockStore.return_value
        handler = FlowTraceHandler(name="test-agent", live=False)
        run_id = uuid.uuid4()
        handler.on_llm_start(
            {"name": "gpt-4o"},
            ["test prompt"],
            run_id=run_id,
        )
        response = MagicMock()
        response.llm_output = {"token_usage": {"total_tokens": 123}}
        handler.on_llm_end(response, run_id=run_id)
        mock_store.save_trace.assert_called_once()
        assert handler.trace is None


def test_llamaindex_handler():
    with patch("flowtrace.llamaindex.SQLiteStore") as MockStore:
        mock_store = MockStore.return_value
        handler = LlamaIndexHandler(name="test-index", live=False)
        event_id = str(uuid.uuid4())
        handler.on_event_start(
            "query", {"query_str": "test query"}, event_id=event_id
        )
        handler.on_event_end(
            "query",
            {"response": "answer"},
            event_id=event_id,
        )
        mock_store.save_trace.assert_called_once()
        assert handler.trace is None


@pytest.mark.skip("async support not yet implemented")
def test_async_context():
    pass  # pragma: no cover


@pytest.mark.skip("replay feature not yet implemented in core")
def test_trace_replay():
    pass  # pragma: no cover
