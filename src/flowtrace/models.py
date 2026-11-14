from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


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

    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        json_encoders={datetime: lambda dt: dt.isoformat() if dt else None},
    )

    @classmethod
    def from_tracer(cls, tracer_step: Any) -> Step:
        """Convert tracer.Step dataclass (and its children) into a Pydantic model.

        Used when persisting or returning traces from a live TraceContext.
        """
        if tracer_step is None:
            return cls(id="missing", name="missing", kind="error")

        children = [cls.from_tracer(child) for child in getattr(tracer_step, "children", [])]
        duration = getattr(tracer_step, "duration", 0.0)

        return cls(
            id=tracer_step.id,
            name=tracer_step.name,
            kind=tracer_step.kind,
            content=getattr(tracer_step, "content", {}).copy(),
            children=children,
            duration_ms=round(duration * 1000, 2),
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
        root_step = Step.from_tracer(context.get_root_step())

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
            start=datetime.now(),
            duration=getattr(context.root_step, "duration", 0.0),
            success=success,
            total_tokens=total_tokens,
        )


class TraceSummary(BaseModel):
    """Lightweight summary for the trace browser / history list."""

    id: str
    name: str
    start: datetime
    duration_ms: float
    success: bool = True
    total_tokens: int = 0

    model_config = ConfigDict(from_attributes=True)
