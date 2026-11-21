from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer


def _uuid_to_hex(uuid_str: str, length: int = 32) -> str:
    """Convert UUID str to hex string (no dashes), sliced to length."""
    return uuid_str.replace("-", "")[:length]


def _to_otel_attribute_value(value: Any) -> Dict[str, Any]:
    """Convert a Python value into an OTEL attribute value object."""
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, dict):
        return {
            "kvlistValue": {
                "values": [
                    {"key": k, "value": _to_otel_attribute_value(v)}
                    for k, v in value.items()
                ]
            }
        }
    if isinstance(value, list):
        return {"arrayValue": {"values": [_to_otel_attribute_value(item) for item in value]}}
    return {"stringValue": str(value)}


class Step(BaseModel):
    """A node in the execution trace.

    Mirrors the structure of tracer.Step.to_dict() for API responses,
    graph rendering and persistence. Fully recursive.
    """

    id: str
    name: str
    kind: str
    content: Dict[str, Any] = Field(default_factory=dict)
    children: List["Step"] = Field(default_factory=list)
    duration_ms: float = 0.0
    start: Optional[datetime] = Field(default=None)

    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
    )

    @field_serializer("start")
    def serialize_start(self, start: Optional[datetime]) -> Optional[str]:
        return start.isoformat() if start else None

    @classmethod
    def from_tracer(
        cls, tracer_step: Any, trace_start: Optional[datetime] = None, root_perf_start: float = 0.0
    ) -> Step:
        """Convert tracer.Step dataclass (and its children) into a Pydantic model.

        If trace_start and root_perf_start are provided, computes wall-clock
        start times for OTEL/LangSmith timestamp accuracy.
        """
        if tracer_step is None:
            return cls(id="missing", name="missing", kind="error")

        children = [
            cls.from_tracer(child, trace_start, root_perf_start)
            for child in getattr(tracer_step, "children", [])
        ]
        duration = getattr(tracer_step, "duration", 0.0)
        perf_start = getattr(tracer_step, "start", root_perf_start)
        start = None
        if trace_start is not None:
            offset = perf_start - root_perf_start
            start = trace_start + timedelta(seconds=offset)

        return cls(
            id=tracer_step.id,
            name=tracer_step.name,
            kind=tracer_step.kind,
            content=getattr(tracer_step, "content", {}).copy(),
            children=children,
            duration_ms=round(duration * 1000, 2),
            start=start,
        )


class Trace(BaseModel):
    """Complete trace model used by storage, API, and the frontend.

    Includes aggregated statistics (tokens, success) derived from the tree.
    """

    id: str
    name: str = "workflow"
    root_step: Step
    start: datetime
    duration: float = 0.0
    success: bool = True
    total_tokens: int = 0

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_tracer_context(cls, context: Any) -> Trace:
        """Build a validated Trace from a live tracer.TraceContext.

        Performs a tree walk to compute aggregate stats. Uses wall-clock
        time for the start field because the tracer only records monotonic
        perf_counter values.
        """
        root_step_obj = getattr(context, "root_step", None)
        trace_start = getattr(context, "start", datetime.now())
        root_perf_start = getattr(root_step_obj, "start", 0.0) if root_step_obj else 0.0
        root_step = Step.from_tracer(root_step_obj, trace_start, root_perf_start)

        total_tokens = 0
        success = True

        def visit(s: Step) -> None:
            nonlocal total_tokens, success
            if s.kind == "llm" and isinstance(s.content, dict):
                total_tokens += int(s.content.get("tokens", 0))
            if isinstance(s.content, dict) and (
                s.content.get("status") == "failed" or "error" in s.content
            ):
                success = False
            for child in s.children:
                visit(child)

        visit(root_step)

        return cls(
            id=context.trace_id,
            name=context.name,
            root_step=root_step,
            start=getattr(context, "start", datetime.now()),
            duration=getattr(context.root_step, "duration", 0.0),
            success=getattr(context, "success", success),
            total_tokens=total_tokens,
        )

    def to_opentelemetry(self) -> Dict[str, Any]:
        """Export to OpenTelemetry compatible OTLP JSON format.

        The returned structure can be sent to an OTEL collector or used with
        Jaeger, Grafana Tempo, or other OTEL-native tools. LangSmith also
        supports OTEL traces.
        """
        trace_id_hex = _uuid_to_hex(self.id, 32)
        spans: List[Dict[str, Any]] = []

        def add_span(step: Step, parent_span_id: Optional[str] = None) -> None:
            span_id_hex = _uuid_to_hex(step.id, 16)
            if step.start is not None:
                start_ns = int(step.start.timestamp() * 1_000_000_000)
            else:
                start_ns = int(self.start.timestamp() * 1_000_000_000)
            duration_ns = int((getattr(step, "duration_ms", 0.0)) * 1_000_000)
            end_ns = start_ns + duration_ns

            attributes: List[Dict[str, Any]] = []
            content = step.content if isinstance(step.content, dict) else {}
            for k, v in content.items():
                if k in (
                    "prompt",
                    "response",
                    "error",
                    "input",
                    "output",
                    "query",
                    "tokens",
                    "status",
                    "invocation_params",
                ):
                    attributes.append(
                        {"key": f"flowtrace.{k}", "value": _to_otel_attribute_value(v)}
                    )
            if step.kind == "llm" and content.get("tokens"):
                attributes.append(
                    {
                        "key": "gen_ai.completion.token_count",
                        "value": _to_otel_attribute_value(content["tokens"]),
                    }
                )

            code = 2 if content.get("status") == "failed" or "error" in content else 0
            status = {"code": code}
            if content.get("error"):
                status["message"] = str(content["error"])

            span = {
                "traceId": trace_id_hex,
                "spanId": span_id_hex,
                "parentSpanId": parent_span_id or "",
                "name": step.name,
                "kind": 3 if step.kind == "llm" else 4 if step.kind in ("tool", "retrieval") else 1,
                "startTimeUnixNano": str(start_ns),
                "endTimeUnixNano": str(end_ns),
                "attributes": attributes,
                "status": status,
            }
            spans.append(span)
            for child in step.children:
                add_span(child, span_id_hex)

        add_span(self.root_step)
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": "flowtrace"}},
                            {"key": "workflow.name", "value": {"stringValue": self.name}},
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "flowtrace", "version": "0.1.0"},
                            "spans": spans,
                        }
                    ],
                }
            ]
        }

    def to_langsmith(self) -> List[Dict[str, Any]]:
        """Export to a list of runs compatible with LangSmith's format.

        Can be used with ``langsmith.Client().run_tree(...)`` or bulk upload.
        """
        runs: List[Dict[str, Any]] = []

        def add_run(step: Step, parent_run_id: Optional[str] = None) -> None:
            content = step.content if isinstance(step.content, dict) else {}
            run_type = {
                "llm": "llm",
                "tool": "tool",
                "retrieval": "tool",
                "embedding": "tool",
                "query": "chain",
                "agent": "chain",
                "chain": "chain",
            }.get(step.kind, "chain")

            inputs = {}
            if "prompt" in content:
                inputs["prompt"] = content["prompt"]
            elif "query" in content:
                inputs["query"] = content["query"]
            elif "input" in content:
                inputs = content["input"] if isinstance(content["input"], dict) else {"input": content["input"]}
            elif "inputs" in content:
                inputs = content["inputs"]

            outputs = {}
            if "response" in content:
                outputs["response"] = content["response"]
            elif "output" in content:
                outputs = content["output"] if isinstance(content["output"], dict) else {"output": content["output"]}
            elif "outputs" in content:
                outputs = content["outputs"]

            error = content.get("error")
            start_time = step.start.isoformat() if step.start else self.start.isoformat()
            end_time = start_time
            if step.start and step.duration_ms > 0:
                dt = timedelta(milliseconds=step.duration_ms)
                end_time = (step.start + dt).isoformat()

            run: Dict[str, Any] = {
                "id": step.id,
                "name": step.name,
                "run_type": run_type,
                "start_time": start_time,
                "end_time": end_time,
                "inputs": inputs,
                "outputs": outputs,
                "error": error,
                "parent_run_id": parent_run_id,
                "trace_id": self.id,
                "extra": {
                    "metadata": {
                        "kind": step.kind,
                        "duration_ms": step.duration_ms,
                    }
                },
            }
            if "tokens" in content:
                run["extra"]["total_tokens"] = content["tokens"]
            runs.append(run)
            for child in step.children:
                add_run(child, step.id)

        add_run(self.root_step)
        return runs


class TraceSummary(BaseModel):
    """Lightweight summary for the trace browser / history list."""

    id: str
    name: str
    start: datetime
    duration_ms: float
    success: bool = True
    total_tokens: int = 0

    model_config = ConfigDict(from_attributes=True)


class TraceComparison(BaseModel):
    """Model returned by the side-by-side comparison endpoint."""

    trace1: Trace
    trace2: Trace
    differences: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


def export_to_opentelemetry(trace: Any) -> Dict[str, Any]:
    """Export any trace (Trace instance or TraceContext) to OpenTelemetry format."""
    if isinstance(trace, Trace):
        return trace.to_opentelemetry()
    # assume it's a TraceContext
    return Trace.from_tracer_context(trace).to_opentelemetry()


def export_to_langsmith(trace: Any) -> List[Dict[str, Any]]:
    """Export any trace (Trace instance or TraceContext) to LangSmith format."""
    if isinstance(trace, Trace):
        return trace.to_langsmith()
    return Trace.from_tracer_context(trace).to_langsmith()
