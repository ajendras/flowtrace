"""Test suite for FlowTrace tracer.

Covers context manager, decorator, recording methods, nesting,
timing, error handling, contextvars and ID uniqueness.
"""

from __future__ import annotations

import time
import uuid

from flowtrace.tracer import TraceContext, current_trace, trace


def test_basic_trace_context():
    """TraceContext initializes root step and supports record_step."""
    with TraceContext(name="basic") as tc:
        tc.record_step("Setup", "init")
        tc.record_step("DefaultKind")  # tests default kind="step"
        steps = tc.root_step.children
        assert len(steps) == 2
        assert steps[0].name == "Setup"
        assert steps[0].kind == "init"
        assert steps[1].name == "DefaultKind"
        assert steps[1].kind == "step"
    assert tc.root_step.duration > 0


def test_nested_steps():
    """Nested steps using context manager and proper stack management."""
    with TraceContext(name="nested") as tc:
        with tc.step("Phase1", "phase"):
            tc.record_step("InnerStep", "sub")
            with tc.step("Deep", "core", {"meta": "value"}) as deep:
                assert deep.kind == "core"
                assert len(tc.step_stack) == 3
    root = tc.root_step
    assert len(root.children) == 1
    phase = root.children[0]
    assert phase.name == "Phase1"
    assert len(phase.children) == 2
    assert phase.children[0].name == "InnerStep"
    assert phase.children[1].name == "Deep"
    assert phase.children[1].content["meta"] == "value"
    assert phase.duration > 0
    assert phase.children[1].duration > 0


def test_decorator_top_level():
    """@trace decorator creates top-level trace when no context active."""
    captured = []

    @trace("TopAgent")
    def top_agent():
        tc = current_trace()
        tc.record_llm_call("hello", "world", 5)
        captured.append(tc)
        return 42

    result = top_agent()
    assert result == 42
    assert len(captured) == 1
    tc = captured[0]
    assert tc.name == "TopAgent"
    assert len(tc.root_step.children) == 1
    wrapper_step = tc.root_step.children[0]
    assert wrapper_step.name == "TopAgent"
    assert wrapper_step.kind == "step"
    assert len(wrapper_step.children) == 1
    llm_step = wrapper_step.children[0]
    assert llm_step.kind == "llm"
    assert llm_step.content["tokens"] == 5


def test_decorator_nested():
    """Decorator nests properly inside existing TraceContext."""
    with TraceContext(name="Parent") as parent_tc:
        @trace("NestedOp", kind="operation")
        def nested_op(x):
            current_trace().record_tool_call("tool1", {"x": x}, "result")
            return x * 2

        res = nested_op(7)
        assert res == 14
    assert len(parent_tc.root_step.children) == 1
    op_step = parent_tc.root_step.children[0]
    assert op_step.name == "NestedOp"
    assert op_step.kind == "operation"
    assert len(op_step.children) == 1
    tool_step = op_step.children[0]
    assert tool_step.kind == "tool"
    assert tool_step.content["name"] == "tool1"


def test_llm_and_tool_recording():
    """LLM and tool recording populate correct step kinds and content."""
    with TraceContext("recorder") as tc:
        tc.record_llm_call(
            prompt="Test prompt",
            response="Test response",
            tokens=42,
            latency=0.123,
        )
        tc.record_tool_call("search", {"q": "test"}, {"results": 3})
    children = tc.root_step.children
    assert len(children) == 2
    llm_steps = [s for s in children if s.kind == "llm"]
    assert len(llm_steps) == 1
    llm = llm_steps[0]
    assert llm.content["prompt"] == "Test prompt"
    assert llm.content["tokens"] == 42
    assert llm.content["latency_ms"] == 123.0
    tool_steps = [s for s in children if s.kind == "tool"]
    assert len(tool_steps) == 1
    tool = tool_steps[0]
    assert tool.name == "search"
    assert tool.content["arguments"]["q"] == "test"
    assert tool.content["output"] == {"results": 3}


def test_timing():
    """Timing is accurately recorded using perf_counter."""
    with TraceContext(name="timer") as tc:
        start = time.perf_counter()
        time.sleep(0.03)
        tc.record_step("Sleeper", "test")
        elapsed = time.perf_counter() - start
    assert tc.root_step.duration >= 0.03
    assert tc.root_step.duration <= elapsed + 0.01


def test_current_trace_contextvar():
    """current_trace() uses ContextVar for correct isolation across contexts."""
    assert current_trace() is None
    with TraceContext(name="ctx1") as tc1:
        assert current_trace() is tc1
        with tc1.step("inner-step", "test"):
            assert current_trace() is tc1
    assert current_trace() is None

    tc_a = None
    tc_b = None
    with TraceContext(name="a") as tc_a:
        tc_a.record_step("a-step", "test")
        with TraceContext(name="b") as tc_b:
            assert current_trace() is tc_b
            tc_b.record_step("b-step", "test")
        assert current_trace() is tc_a
    assert current_trace() is None
    assert tc_a is not None
    assert tc_b is not None


def test_error_handling_in_context():
    """Exceptions are recorded in the trace tree."""
    tc = None
    try:
        with TraceContext(name="error-demo") as tc:
            tc.record_step("Before")  # uses default kind="step"
            1 / 0  # raise ZeroDivisionError
    except ZeroDivisionError:
        pass  # __exit__ already recorded it
    assert tc is not None
    root = tc.get_root_step()
    error_steps = [s for s in root.children if s.name == "Execution Error"]
    assert len(error_steps) == 1
    error_step = error_steps[0]
    assert error_step.kind == "error"
    assert error_step.content["error_type"] == "ZeroDivisionError"
    assert "division by zero" in error_step.content["error_message"].lower()
    assert error_step.content["status"] == "failed"


def test_unique_ids():
    """Each trace and step gets unique UUIDs."""
    ids = set()
    for i in range(3):
        with TraceContext(f"trace-{i}") as tc:
            ids.add(tc.trace_id)
            tc.record_step("s1", "test")
            tc.record_step("s2", "test")
            for child in tc.root_step.children:
                ids.add(child.id)
    assert len(ids) == 9  # 3 traces + 6 steps
    for iid in ids:
        uuid.UUID(iid)
