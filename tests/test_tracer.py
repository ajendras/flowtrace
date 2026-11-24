"""Tests for the FlowTrace core tracer."""

import pytest
from unittest.mock import patch, MagicMock

from flowtrace.tracer import TraceContext, current_trace, trace
from flowtrace.models import Trace
from flowtrace.storage import SQLiteStore


def test_tracecontext_step_annotation():
    with TraceContext(name="AnnotationTest") as tc:
        with tc.step("TestStep", kind="phase") as step:
            step.annotate("This step is critical for debugging")
            step.annotate("Second note", {"priority": "high"})
        tc.record_step("AnotherStep").annotate("Simple step comment")

    assert len(tc.root_step.children) == 2
    assert len(tc.root_step.children[0].annotations) == 2
    assert tc.root_step.children[0].annotations[0]["text"] == "This step is critical for debugging"
    assert tc.root_step.children[0].annotations[1]["metadata"]["priority"] == "high"
    assert len(tc.root_step.children[1].annotations) == 1
    assert tc.root_step.children[1].annotations[0]["text"] == "Simple step comment"


def test_tracecontext_annotate_by_id():
    with TraceContext(name="IDAnnoTest") as tc:
        s1 = tc.record_step("StepOne")
        tc.annotate("Comment via ID", s1.id)
        s2 = tc.record_step("StepTwo")
        tc.annotate("Another comment", s2.id, {"author": "tester"})

    assert len(s1.annotations) == 1
    assert s1.annotations[0]["text"] == "Comment via ID"
    assert len(s2.annotations) == 1
    assert s2.annotations[0]["metadata"]["author"] == "tester"


def test_annotation_in_pydantic_model():
    with TraceContext(name="ModelTest") as tc:
        with tc.step("Annotated") as s:
            s.annotate("Model conversion test")
            s.annotate("Second annotation for model")

    trace = Trace.from_tracer_context(tc)
    assert len(trace.root_step.children) == 1
    child = trace.root_step.children[0]
    assert len(child.annotations) == 2
    assert child.annotations[0]["text"] == "Model conversion test"
    assert child.annotations[1]["text"] == "Second annotation for model"


def test_trace_decorator_with_annotation():
    @trace("DecoratedFunc")
    def func():
        tc = current_trace()
        assert tc is not None
        with tc.step("InsideDecorator") as step:
            step.annotate("Inside decorated function")
        return "done"

    result = func()
    assert result == "done"


@pytest.mark.asyncio
async def test_live_trace_with_annotation():
    with patch("httpx.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=202)
        with TraceContext(name="LiveAnno", live=True) as tc:
            with tc.step("LiveStep") as step:
                step.annotate("Live annotation should trigger update")
        assert mock_post.called


def test_step_count_with_annotations():
    with TraceContext("count_test") as tc:
        tc.record_step("s1")
        with tc.step("s2") as s2:
            s2.annotate("note1")
            s2.annotate("note2")

    t = Trace.from_tracer_context(tc)
    assert len(t.root_step.children) == 2
    assert len(t.root_step.children[1].annotations) == 2
    assert t.root_step.children[1].annotations[0]["text"] == "note1"


def test_annotate_on_current_step():
    with TraceContext(name="CurrentTest") as tc:
        tc.record_step("setup")
        with tc.step("current") as _:
            tc.annotate("Current step comment")
    assert len(tc.root_step.children[1].annotations) == 1
    assert tc.root_step.children[1].annotations[0]["text"] == "Current step comment"
