import time
import uuid
import pytest
from unittest.mock import MagicMock

from flowtrace.tracer import TraceContext, trace, current_trace
from flowtrace import FlowTraceHandler, LlamaIndexHandler


def test_basic_trace_context():
    with TraceContext(name="BasicTest") as tc:
        assert tc.name == "BasicTest"
        assert tc.root_step.name == "BasicTest"
        assert tc.root_step.kind == "root"
        tc.record_llm_call("test prompt", "test response", tokens=42)
        assert len(tc.root_step.children) == 1
        assert tc.root_step.children[0].kind == "llm"
        assert tc.root_step.children[0].content["tokens"] == 42
        assert tc.total_tokens == 42
    assert tc.duration > 0
    assert tc.success is True


def test_nested_steps():
    with TraceContext(name="Nested") as tc:
        with tc.step("Parent", kind="phase") as parent:
            with tc.step("Child", kind="sub") as child:
                child.annotate("child detail")
            tc.record_tool_call("tool", {"q": "x"}, "result")
        assert len(tc.root_step.children) == 1
        assert len(parent.children) == 2
        assert parent.children[0].name == "Child"
        assert parent.children[1].kind == "tool"


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
        assert len(phase.children) == 2

        llm = next(c for c in phase.children if c.kind == "llm")
        sub = next(c for c in phase.children if c.name == "Substep")
        assert llm.content["prompt"] == "prompt1"
        assert llm.content["response"] == "resp1"
        assert len(llm.annotations) == 0
        assert len(sub.annotations) == 1
        assert sub.annotations[0]["metadata"]["key"] == "value"


def test_record_step_and_annotate():
    with TraceContext(name="AnnotTest") as tc:
        step = tc.record_step("LogEvent", kind="event", metadata={"id": 123})
        step.annotate("important note", {"priority": "high"})
        assert len(tc.root_step.children) == 1
        assert len(step.annotations) == 1
        assert step.annotations[0]["text"] == "important note"
        assert step.annotations[0]["metadata"]["priority"] == "high"


def test_decorator():
    @trace("DecoratedFunc")
    def sample():
        tc = current_trace()
        tc.record_llm_call("hi", "hello", 3)
        return "done"

    result = sample()
    assert result == "done"


def test_current_trace():
    with TraceContext(name="CurrentTest") as tc:
        got = current_trace()
        assert got is tc
        got.record_tool_call("search", {"query": "foo"}, "bar")
        assert len(tc.root_step.children) == 1


def test_error_capture():
    with pytest.raises(ValueError, match="boom"):
        with TraceContext(name="ErrorTest") as tc:
            with tc.step("Risky") as s:
                s.annotate("about to fail")
                raise ValueError("boom")
    assert tc.success is False
    assert len(tc.root_step.children) == 1
    risky = tc.root_step.children[0]
    assert risky.name == "Risky"
    assert risky.content.get("status") == "failed"
    assert "error" in risky.content


def test_timing_accuracy():
    with TraceContext(name="Timing") as tc:
        with tc.step("SlowStep") as s:
            time.sleep(0.05)
    assert s.duration >= 0.04


def test_unique_ids():
    ids = set()
    for _ in range(50):
        with TraceContext(name="Unique") as tc:
            ids.add(tc.id)
            with tc.step("s1") as s:
                ids.add(s.id)
    assert len(ids) == 100


def test_flowtrace_handler():
    handler = FlowTraceHandler(name="TestAgent", live=False)
    run_id = uuid.uuid4()
    handler.on_chain_start(
        {"name": "TestChain"}, {"input": "hello"}, run_id=run_id
    )
    llm_id = uuid.uuid4()
    handler.on_llm_start(
        {"name": "TestLLM"},
        ["prompt"],
        run_id=llm_id,
        parent_run_id=run_id,
    )
    mock_response = MagicMock()
    mock_response.llm_output = {"token_usage": {"total_tokens": 25}}
    mock_response.generations = []
    handler.on_llm_end(mock_response, run_id=llm_id)
    handler.on_chain_end({"output": "final"}, run_id=run_id)
    assert handler.trace is None
    assert handler.root_run_id is None


def test_llamaindex_handler():
    handler = LlamaIndexHandler(name="TestIndex", live=False)
    event_id = "evt-123"
    handler.on_event_start("QUERY", {"query_str": "test query"}, event_id=event_id)
    handler.on_event_end(
        "QUERY",
        {"response": "answer", "total_tokens": 42},
        event_id=event_id,
    )
    assert handler.trace is None


def test_annotate_via_trace():
    with TraceContext(name="AnnotViaTc") as tc:
        tc.record_step("Step1")
        tc.annotate("global note", metadata={"source": "test"})
        assert len(tc.root_step.annotations) >= 1
