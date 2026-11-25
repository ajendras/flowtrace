"""Core tracing functionality for FlowTrace.

Provides TraceContext for manual instrumentation, the @trace decorator,
current_trace() contextvar, and Step tree construction with support for
annotations, LLM calls, tool calls and live streaming.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


current_trace_var: ContextVar[Optional[TraceContext]] = ContextVar(
    "current_trace", default=None
)


@dataclass
class Step:
    """Runtime step for building the trace tree during execution."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    kind: str = "step"
    content: Dict[str, Any] = field(default_factory=dict)
    start: float = field(default_factory=time.perf_counter)
    duration: float = 0.0
    children: list[Step] = field(default_factory=list)
    annotations: list[Dict[str, Any]] = field(default_factory=list)

    def annotate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an annotation to this step."""
        self.annotations.append(
            {
                "text": text,
                "timestamp": time.time(),
                "metadata": metadata or {},
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert step (and children) to dict for serialization/live updates."""
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "content": self.content.copy(),
            "duration": round(self.duration, 4),
            "children": [c.to_dict() for c in self.children],
            "annotations": list(self.annotations),
        }


class TraceContext:
    """Main context for a trace. Manages the step tree, timing, and live updates."""

    def __init__(self, name: str = "workflow", live: bool = False) -> None:
        self.id: str = str(uuid.uuid4())
        self.name: str = name
        self.live: bool = live
        self.root_step: Step = Step(name=name, kind="root", start=time.perf_counter())
        self.current_step: Step = self.root_step
        self.start: float = time.perf_counter()
        self.duration: float = 0.0
        self.success: bool = True
        self.total_tokens: int = 0
        self._token: Any = None
        self._server_url: Optional[str] = None

    def __enter__(self) -> TraceContext:
        self._token = current_trace_var.set(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.duration = time.perf_counter() - self.start
        if exc_type is not None:
            self.success = False
        if self.live:
            self._send_live_update()
        if self._token is not None:
            current_trace_var.reset(self._token)
        return False

    def _send_live_update(self) -> None:
        """Send partial trace to the server for live UI updates (best-effort)."""
        if not self._server_url:
            import os

            self._server_url = os.getenv(
                "FLOWTRACE_SERVER_URL", "http://127.0.0.1:8080"
            )
        try:
            import httpx

            payload = {
                "name": self.name,
                "root_step": self.root_step.to_dict(),
                "start": self.start,
                "duration": self.duration,
                "success": self.success,
                "total_tokens": self.total_tokens,
            }
            httpx.post(
                f"{self._server_url}/api/traces/{self.id}/live",
                json=payload,
                timeout=0.5,
            )
        except Exception:
            pass  # live updates are best-effort

    @contextmanager
    def step(
        self,
        name: str,
        kind: str = "step",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Generator[Step, None, None]:
        """Context manager for a nested step."""
        step = Step(
            name=name,
            kind=kind,
            content=metadata or {},
            start=time.perf_counter(),
        )
        self.current_step.children.append(step)
        previous = self.current_step
        self.current_step = step
        try:
            yield step
        except Exception as e:
            step.content.update({"error": str(e), "status": "failed"})
            raise
        finally:
            step.duration = time.perf_counter() - step.start
            self.current_step = previous
            if self.live:
                self._send_live_update()

    def record_step(
        self,
        name: str,
        kind: str = "step",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Step:
        """Record a non-nested step and return it for annotation."""
        step = Step(
            name=name,
            kind=kind,
            content=metadata or {},
            start=time.perf_counter(),
        )
        self.current_step.children.append(step)
        step.duration = 0.0
        if self.live:
            self._send_live_update()
        return step

    def record_llm_call(
        self,
        prompt: str,
        response: str = "",
        tokens: int = 0,
        **kwargs: Any,
    ) -> None:
        """Record an LLM call as a step."""
        content: Dict[str, Any] = {
            "prompt": prompt,
            "response": response,
            "tokens": tokens,
            **kwargs,
        }
        if "status" not in content:
            content["status"] = "completed"
        step = Step(
            name="LLM Call",
            kind="llm",
            content=content,
            start=time.perf_counter(),
        )
        self.current_step.children.append(step)
        step.duration = kwargs.get("latency", 0.1)
        self.total_tokens += tokens
        if self.live:
            self._send_live_update()

    def record_tool_call(
        self,
        name: str,
        args: Dict[str, Any],
        result: Any,
        **kwargs: Any,
    ) -> None:
        """Record a tool call as a step."""
        content: Dict[str, Any] = {"args": args, "result": result, **kwargs}
        if "status" not in content:
            content["status"] = "completed"
        step = Step(
            name=name,
            kind="tool",
            content=content,
            start=time.perf_counter(),
        )
        self.current_step.children.append(step)
        step.duration = kwargs.get("latency", 0.05)
        if self.live:
            self._send_live_update()

    def annotate(
        self,
        text: str,
        step_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add annotation to current step or a specific step by ID."""
        if step_id is None:
            target = self.current_step
        else:
            target = self._find_step_by_id(self.root_step, step_id)
            if target is None:
                return False
        target.annotate(text, metadata)
        if self.live:
            self._send_live_update()
        return True

    def _find_step_by_id(self, step: Step, step_id: str) -> Optional[Step]:
        if step.id == step_id:
            return step
        for child in step.children:
            found = self._find_step_by_id(child, step_id)
            if found is not None:
                return found
        return None


def current_trace() -> TraceContext:
    """Return the active trace context for the current contextvar."""
    trace = current_trace_var.get()
    if trace is None:
        raise RuntimeError(
            "No active TraceContext. Use 'with TraceContext()' or the @trace decorator."
        )
    return trace


def trace(name: Optional[str] = None) -> Callable[[Callable], Callable]:
    """Decorator for tracing a function.

    Can be used as ``@trace`` or ``@trace("CustomName")``.
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tname = name or func.__name__
            with TraceContext(name=tname) as tc:
                return func(*args, **kwargs)

        return wrapper

    if callable(name):  # used as @trace without ()
        func = name
        return decorator(func)
    return decorator
