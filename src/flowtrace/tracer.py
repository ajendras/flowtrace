"""Core tracing functionality for FlowTrace.

Provides TraceContext for manual instrumentation, the @trace decorator,
current_trace() contextvar accessor, and the Step tree builder.
Supports live streaming to the dashboard via HTTP.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, Optional

current: ContextVar[Optional[TraceContext]] = ContextVar(
    "current_trace", default=None
)


def current_trace() -> TraceContext:
    """Return the active TraceContext or raise if none."""
    ctx = current.get()
    if ctx is None:
        raise RuntimeError(
            "No active trace. Use `with TraceContext():` or the @trace decorator."
        )
    return ctx


class Step:
    """A step in the trace tree. Mutable during construction."""

    def __init__(
        self,
        id: Optional[str] = None,
        name: str = "step",
        kind: str = "step",
        content: Optional[Dict[str, Any]] = None,
        start: Optional[float] = None,
    ) -> None:
        self.id = id or str(uuid.uuid4())
        self.name = name
        self.kind = kind
        self.content = content or {}
        self.children: list[Step] = []
        self.start = start or time.perf_counter()
        self.duration: float = 0.0

    def __repr__(self) -> str:
        return (
            f"Step(id={self.id!r}, name={self.name!r}, kind={self.kind!r}, "
            f"content={self.content!r}, children={len(self.children)}, "
            f"start={self.start:.9f}, duration={self.duration})"
        )


class TraceContext:
    """Context manager for tracing a workflow or agent run.

    Builds a nested tree of Steps using perf_counter for accurate timing.
    Supports live streaming to the FlowTrace dashboard.
    """

    def __init__(self, name: str = "Workflow", live: bool = False) -> None:
        self.name = name
        self.trace_id = str(uuid.uuid4())
        self.live = live
        self.start = datetime.now()
        self.root_step = Step(
            id=self.trace_id,
            name=name,
            kind="chain",
            content={},
            start=time.perf_counter(),
        )
        self.success = True
        self.total_tokens = 0
        self._token: Optional[object] = None
        self._step_stack: list[Step] = [self.root_step]

    def __enter__(self) -> TraceContext:
        self._token = current.set(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool | None:
        if exc_type is not None:
            self.success = False
            self.root_step.content.setdefault("error", str(exc_val))
            self.root_step.content["status"] = "failed"
        self.root_step.duration = time.perf_counter() - self.root_step.start
        if self.live:
            self._broadcast_live()
        if self._token is not None:
            current.reset(self._token)
        return False

    @contextmanager
    def step(
        self, name: str, kind: str = "step", **metadata: Any
    ) -> Iterator[Step]:
        """Context manager for a nested step."""
        step = Step(
            name=name,
            kind=kind,
            content=dict(metadata),
            start=time.perf_counter(),
        )
        parent = self._step_stack[-1]
        parent.children.append(step)
        self._step_stack.append(step)
        try:
            yield step
        except Exception as e:
            step.content["error"] = str(e)
            step.content["status"] = "failed"
            self.success = False
            raise
        else:
            step.content.setdefault("status", "completed")
        finally:
            step.duration = time.perf_counter() - step.start
            self._step_stack.pop()

    def record_step(self, name: str, kind: str = "step", **metadata: Any) -> None:
        """Record an instantaneous step (duration = 0)."""
        step = Step(
            name=name,
            kind=kind,
            content=dict(metadata),
            start=time.perf_counter(),
        )
        parent = self._step_stack[-1]
        parent.children.append(step)
        # duration remains 0.0 for instantaneous records

    def record_llm_call(
        self,
        prompt: str,
        response: str | None = None,
        tokens: int = 0,
        latency: float = 0.0,
        **extra: Any,
    ) -> None:
        """Record an LLM call as a step."""
        content: Dict[str, Any] = {
            "prompt": prompt,
            "status": "completed",
            "tokens": tokens,
        }
        if response:
            content["response"] = response
        content.update(extra)
        step = Step(
            name="LLM",
            kind="llm",
            content=content,
            start=time.perf_counter() - latency,
        )
        step.duration = latency
        self._step_stack[-1].children.append(step)
        self.total_tokens += tokens

    def record_tool_call(
        self,
        name: str,
        args: Dict[str, Any] | None = None,
        result: Any = None,
        latency: float = 0.0,
        **extra: Any,
    ) -> None:
        """Record a tool invocation as a step."""
        content: Dict[str, Any] = {"status": "completed"}
        if args:
            content["input"] = args
        if result is not None:
            content["output"] = result
        content.update(extra)
        step = Step(
            name=name,
            kind="tool",
            content=content,
            start=time.perf_counter() - latency,
        )
        step.duration = latency or 0.0
        self._step_stack[-1].children.append(step)

    def _broadcast_live(self) -> None:
        """Send live update to the dashboard server via HTTP."""
        server_url = os.getenv("FLOWTRACE_SERVER_URL", "http://127.0.0.1:8080")
        try:
            import httpx

            from .models import Trace

            trace_model = Trace.from_tracer_context(self)
            payload = trace_model.model_dump(mode="json")
            httpx.post(
                f"{server_url}/api/traces/{self.trace_id}/live",
                json=payload,
                timeout=5.0,
            )
        except Exception:
            # Live updates are best-effort
            pass


def trace(name: str = "Agent") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to trace a function with a TraceContext."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with TraceContext(name=name, live=False) as tc:
                return func(*args, **kwargs)

        return wrapper

    return decorator
