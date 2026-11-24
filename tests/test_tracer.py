"""Tests for the FlowTrace core tracer.

Covers the dataclass ordering fix (root cause of the 3.11 CI failure),
context management, decorator, nesting, recording helpers, annotations,
error capture, contextvar semantics, and UUID uniqueness.
"""

import time
import uuid

import pytest

from flowtrace.tracer import TraceContext, current_trace, trace, Step
from flowtrace.models import Trace
from flowtrace.storage import SQLiteStore


def test_step_dataclass_field_order():
    """Regression test for the CI failure.

    Non-default argument 'name' followed a default argument in the
    dataclass definition, which dataclasses.py rejects on Python 3.11.
    Reordering fields (name first) fixes it while preserving all
    existing behaviour and constructor usage.
    """
    # Would have raised TypeError: non-default argument 'name' follows default argument
    step = Step(name="test-step", kind="llm", content={"prompt": "hi"})
    assert step.name == "test-step"
    assert step.kind == "llm"
    assert len(step.id) >= 10
    assert step.duration == 0.0
    assert len(step.children) == 0
    assert len(step.annotations) == 0


def test_trace_context_basic():
    store = SQLiteStore()
    store.clear_all()  # isolate test

    with TraceContext(name="TestWorkflow") as tc:
        assert current_trace() is tc
        assert tc.name == "TestWorkflow"
        assert tc.success is True

        tc.record_llm_call("What is observability?", "Essential for agents", tokens=42)
        tc.record_tool_call("search", {"q": "flowtrace"}, {"status": "ok"})

        assert len(tc.root_step.children) == 2
        assert tc.root_step.children[0].kind == "llm"
        assert tc.root_step.children[0].content["tokens"] == 42
        assert tc.root_step.children[1].kind == "tool"

    assert tc.duration > 0.0
    assert tc.success is True

    # persisted
    saved = store.get_trace(tc.id)
    assert saved is not None
    assert saved.name == "TestWorkflow"
    assert saved.total_tokens == 42


def test_nested_steps_and_annotate():
    with TraceContext(name="NestedTest") as tc:
        with tc.step("Phase1", kind="phase", metadata={"version": "2"}) as s1:
            s1.annotate("Phase started", {"user": "test"})
            tc.record_llm_call("prompt1", "resp1", tokens=10)

            with tc.step("Substep", kind="sub") as s2:
                s2.annotate("nested detail", {"key": "value"})
                assert len(s2.annotations) == 1

        assert len(tc.root_step.children) == 1
        phase = tc.root_step.children[0]
        assert phase.kind == "phase"
        assert phase.content["version"] == "2"
        assert len(phase.annotations) == 1
        assert phase.annotations[0]["metadata"]["user"] == "test"
        assert len(phase.children) == 1
        assert phase.children[0].annotations[0]["metadata"]["key"] == "value"

    assert tc.success is True


def test_decorator():
    calls = []

    @trace("DecoratedAgent")
    def agent_task(x: int) -> int:
        tc = current_trace()
        assert tc is not None
        assert tc.name == "DecoratedAgent"
        tc.record_step("inner-step")
        calls.append(x)
        return x * 2

    result = agent_task(21)
    assert result == 42
    assert len(calls) == 1


def test_bare_decorator():
    @trace
    def bare_func():
        tc = current_trace()
        assert tc.name == "bare_func"
        return "ok"

    assert bare_func() == "ok"


def test_error_capture():
    with pytest.raises(ValueError, match="boom"):
        with TraceContext(name="ErrorTrace") as tc:
            tc.record_tool_call("failing-tool", {}, "pre-error")
            raise ValueError("boom")

    assert tc.success is False
    assert "error" in tc.root_step.content
    assert tc.root_step.content.get("status") == "failed"


def test_current_trace_contextvar_isolation():
    outer = TraceContext("outer")
    outer.__enter__()
    assert current_trace() is outer

    with TraceContext("inner") as inner:
        assert current_trace() is inner
        inner.record_llm_call("inner prompt", "inner resp")

    assert current_trace() is outer
    outer.__exit__(None, None, None)
    assert current_trace() is None


def test_step_annotate_and_tc_annotate():
    with TraceContext("AnnotateTest") as tc:
        step = tc.record_step("manual-step")
        step.annotate("direct on step")

        tc.annotate("on root")
        tc.annotate("targeted", step_id=step.id, metadata={"source": "test"})

        assert len(step.annotations) == 2
        assert len(tc.root_step.annotations) == 1
        assert any(a["metadata"].get("source") == "test" for a in step.annotations)


def test_live_flag_does_not_raise():
    """live=True should not raise even if server is unreachable."""
    with TraceContext(name="LiveTest", live=True) as tc:
        tc.record_llm_call("live prompt", "live response", tokens=5)
        # _broadcast_live is best-effort (httpx failure is swallowed)
    assert tc.live is True
    assert tc.success is True


def test_uuid_uniqueness():
    ids = {TraceContext().id for _ in range(50)}
    assert len(ids) == 50


def test_compatibility_with_handler_style_step_creation():
    """Ensure handlers that build Steps directly continue to work."""
    tc = TraceContext("handler-test")
    tc.__enter__()

    # LangChain/LlamaIndex style of direct Step construction with explicit id
    step = Step(
        id=str(uuid.uuid4()),
        name="HandlerLLM",
        kind="llm",
        content={"prompt": "handler prompt"},
        start=time.perf_counter(),
    )
    tc.root_step.children.append(step)
    step.duration = 0.123

    tc.__exit__(None, None, None)
    assert len(tc.root_step.children) == 1
    assert tc.root_step.children[0].name == "HandlerLLM"
