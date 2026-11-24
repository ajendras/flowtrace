from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer
import uuid


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

    Supports manual annotations/comments via ``annotate()`` on the tracer Step
    or via the ``/annotate`` API endpoint.
    """
    id: str
    name: str
    kind: str = "step"
    content: Dict[str, Any] = Field(default_factory=dict)
    duration: float = 0.0
    children: List["Step"] = Field(default_factory=list)
    annotations: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
    )

    @classmethod
    def from_tracer_step(cls, step: Any) -> "Step":
        """Convert a tracer.py dataclass Step into a Pydantic model."""
        if isinstance(step, dict):
            return cls.model_validate(step)
        return cls(
            id=step.id,
            name=step.name,
            kind=step.kind,
            content=step.content,
            duration=getattr(step, "duration", 0.0),
            children=[cls.from_tracer_step(child) for child in getattr(step, "children", [])],
            annotations=getattr(step, "annotations", []),
        )


class Trace(BaseModel):
    """Complete trace with summary fields for fast listing and analytics."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    start: datetime = Field(default_factory=datetime.utcnow)
    duration: float = 0.0
    success: bool = True
    total_tokens: int = 0
    root_step: Step

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_tracer_context(cls, tc: Any) -> "Trace":
        """Build a Pydantic Trace from a live TraceContext (avoids circular import)."""
        return cls(
            id=getattr(tc, "id", str(uuid.uuid4())),
            name=getattr(tc, "name", "workflow"),
            start=datetime.utcnow(),
            duration=getattr(tc, "duration", 0.0),
            success=getattr(tc, "success", True),
            total_tokens=getattr(tc, "total_tokens", 0),
            root_step=Step.from_tracer_step(getattr(tc, "root_step", None)),
        )

    @property
    def duration_ms(self) -> float:
        return round(self.duration * 1000, 2)

    def get_flame_data(self) -> List[Dict[str, Any]]:
        """Return flat list of steps with depth for flamegraph rendering."""
        data = []
        def visit(s: Step, depth: int = 0, offset: float = 0.0) -> float:
            start = offset
            for child in s.children:
                offset = visit(child, depth + 1, offset)
            data.append(
                {
                    "id": s.id,
                    "name": s.name,
                    "kind": s.kind,
                    "duration": s.duration,
                    "depth": depth,
                    "start": start,
                    "annotations": s.annotations,
                }
            )
            return offset + s.duration
        visit(self.root_step)
        return data

    def get_token_data(self) -> List[Dict[str, Any]]:
        """Extract LLM steps for token bar chart."""
        tokens = []
        def visit(s: Step):
            if s.kind == "llm" and s.content.get("tokens"):
                tokens.append(
                    {
                        "name": s.name,
                        "tokens": s.content.get("tokens", 0),
                        "annotations": len(s.annotations),
                    }
                )
            for c in s.children:
                visit(c)
        visit(self.root_step)
        return sorted(tokens, key=lambda x: x["tokens"], reverse=True)

    def add_annotation(
        self, step_id: str, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add an annotation to a step by ID (used by API and storage)."""
        def recurse(step: Step) -> bool:
            if step.id == step_id:
                step.annotations.append(
                    {
                        "text": text,
                        "timestamp": datetime.utcnow().isoformat(),
                        "metadata": metadata or {},
                    }
                )
                return True
            for child in step.children:
                if recurse(child):
                    return True
            return False
        return recurse(self.root_step)

    def to_opentelemetry(self) -> Dict[str, Any]:
        """Export to OTLP format (resourceSpans)."""
        return {
            "resourceSpans": [
                {
                    "resource": {"attributes": []},
                    "scopeSpans": [
                        {
                            "scope": {},
                            "spans": self._build_otel_spans(),
                        }
                    ],
                }
            ]
        }

    def _build_otel_spans(self) -> List[Dict]:
        spans = []
        def recurse(s: Step, parent_id: Optional[str] = None):
            span_id = _uuid_to_hex(s.id, 16)
            trace_id = _uuid_to_hex(self.id, 32)
            span = {
                "traceId": trace_id,
                "spanId": span_id,
                "name": s.name,
                "kind": 1,
                "startTimeUnixNano": "0",
                "endTimeUnixNano": "0",
                "attributes": [
                    {"key": "flowtrace.kind", "value": _to_otel_attribute_value(s.kind)},
                    {"key": "flowtrace.annotations", "value": _to_otel_attribute_value(s.annotations)},
                ],
                "status": {},
            }
            if parent_id:
                span["parentSpanId"] = parent_id
            spans.append(span)
            for child in s.children:
                recurse(child, span_id)
        recurse(self.root_step)
        return spans

    def to_langsmith(self) -> List[Dict]:
        """Export to LangSmith run format (flat list of runs)."""
        runs = []
        def recurse(s: Step, parent_run_id: Optional[str] = None):
            run = {
                "id": s.id,
                "name": s.name,
                "run_type": s.kind,
                "inputs": s.content.get("prompt") or s.content.get("args"),
                "outputs": s.content.get("response") or s.content.get("result"),
                "parent_run_id": parent_run_id,
                "trace_id": self.id,
                "extra": {"annotations": s.annotations},
            }
            runs.append(run)
            for child in s.children:
                recurse(child, s.id)
        recurse(self.root_step)
        return runs


class TraceSummary(BaseModel):
    """Lightweight summary for the trace list view."""
    id: str
    name: str
    start: datetime
    duration_ms: float
    total_tokens: int
    success: bool
    step_count: int


class TraceComparison(BaseModel):
    """Result of side-by-side trace comparison."""
    trace1: Trace
    trace2: Trace
    differences: Dict[str, Any]
