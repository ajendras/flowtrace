"""Core tracing functionality for FlowTrace.

Provides the TraceContext context manager, the @trace decorator, Step tree construction,
and the current_trace() lookup. Supports live streaming to the dashboard.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional
from uuid import uuid4

from .models import Trace


current_trace_var: ContextVar[Optional[TraceContext]] = ContextVar(
    "current_trace_var", default=None
)


@dataclass
class Step:
    """Represents a single step in the trace tree."""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str
    kind: str = "step"
    content: Dict[str, Any] = field(default_factory=dict)
    start: float = field(default_factory=time.perf_counter)
    duration: float = 0.0
    children: List[Step] = field(default_factory=list)
    annotations: List[Dict[str, Any]] = field(default_factory=list)

    def annotate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a manual annotation or comment to this step.

        These appear in the UI inspector and are persisted with the trace.
        """
        self.annotations.append(
            {
                "text": text,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {},
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization and live updates."""
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "content": self.content,
            "duration": round(self.duration, 4),
            "children": [child.to_dict() for child in self.children],
            "annotations": self.annotations,
        }


class TraceContext:
    """Main context for building a trace with nested steps.

    Can be used as context manager or via the @trace decorator.
    """

    def __init__(self, name: str = "workflow", live: bool = False) -> None:
        self.id = str(uuid4())
        self.name = name
        self.live = live
        self.root_step = Step(name=name, kind="root")
        self.start = time.perf_counter()
        self.duration = 0.0
        self.success = True
        self.total_tokens = 0
        self._current_step: Step = self.root_step
        self._token: Optional[Any] = None

    def __enter__(self) -> TraceContext:
        self._token = current_trace_var.set(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.duration = time.perf_counter() - self.start
        if exc_type is not None:
            self.success = False
            self.root_step.content.setdefault("error", str(exc_val))
            self.root_step.content["status"] = "failed"
        if self.live:
            self._send_live_update()
        if self._token is not None:
            current_trace_var.reset(self._token)

    def _send_live_update(self) -> None:
        try:
            import httpx
            server_url = os.getenv("FLOWTRACE_SERVER_URL", "http://127.0.0.1:8080")
            payload = {
                "name": self.name,
                "root_step": self.root_step.to_dict(),
                "start": datetime.utcnow().isoformat(),
                "duration": round(self.duration, 4),
                "success": self.success,
                "total_tokens": self.total_tokens,
            }
            httpx.post(
                f"{server_url}/api/traces/{self.id}/live",
                json=payload,
                timeout=5.0,
            )
        except Exception:
            # Live update failure should not break the trace
            pass

    @contextmanager
    def step(
        self,
        name: str,
        kind: str = "step",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Generator[Step, None, None]:
        """Context manager for a nested step. Yields the Step so it can be annotated."""
        content = metadata or {}
        step = Step(name=name, kind=kind, content=content)
        self._current_step.children.append(step)
        previous = self._current_step
        self._current_step = step
        try:
            yield step
        finally:
            step.duration = time.perf_counter() - step.start
            self._current_step = previous

    def record_step(
        self, name: str, kind: str = "step", **kwargs: Any
    ) -> Step:
        """Record a completed step with optional content."""
        step = Step(name=name, kind=kind, content=kwargs)
        step.duration = 0.001  # placeholder, can be updated
        self._current_step.children.append(step)
        return step

    def record_llm_call(
        self,
        prompt: str,
        response: str,
        tokens: int = 0,
        latency: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Record an LLM call as a step."""
        content: Dict[str, Any] = {
            "prompt": prompt,
            "response": response,
            "tokens": tokens,
            "status": "completed",
            **kwargs,
        }
        step = Step(
            name="LLM Call",
            kind="llm",
            content=content,
            duration=latency,
        )
        self._current_step.children.append(step)
        self.total_tokens += tokens

    def record_tool_call(
        self,
        name: str,
        args: Dict[str, Any],
        result: Any,
        **kwargs: Any,
    ) -> None:
        """Record a tool invocation as a step."""
        content: Dict[str, Any] = {
            "args": args,
            "result": result,
            "status": "completed",
            **kwargs,
        }
        step = Step(name=name, kind="tool", content=content)
        self._current_step.children.append(step)

    def annotate(
        self, text: str, step_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add annotation to a step by id (or current step if omitted)."""
        if step_id is None:
            target = self._current_step
        else:
            target = self._find_step(self.root_step, step_id)
            if target is None:
                return
        target.annotate(text, metadata)

    def _find_step(self, step: Step, sid: str) -> Optional[Step]:
        if step.id == sid:
            return step
        for child in step.children:
            found = self._find_step(child, sid)
            if found:
                return found
        return None

    def get_trace(self) -> Trace:
        """Convert this context to a Pydantic Trace model."""
        return Trace.from_tracer_context(self)


def trace(name: Optional[str] = None) -> Callable[[Callable], Callable]:
    """Decorator to trace a function."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with TraceContext(
                name=name or func.__name__, live=False
            ) as tc:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def current_trace() -> Optional[TraceContext]:
    """Get the current trace context from the ContextVar."""
    return current_trace_var.get()
