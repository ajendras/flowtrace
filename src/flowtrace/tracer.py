"""Core tracer for FlowTrace.

TraceContext is a context manager that builds a nested tree of steps.
Supports manual recording, context managers for steps, the @trace decorator,
and live streaming to the local dashboard via httpx.

Uses contextvars for async/thread safety of current_trace().
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar, cast

import httpx

_current_trace: ContextVar[Optional["TraceContext"]] = ContextVar(
    "current_trace", default=None
)

T = TypeVar("T", bound=Callable[..., Any])


@dataclass
class Step:
    """A node in the trace tree (dataclass used by TraceContext)."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    kind: str = "step"
    content: Dict[str, Any] = field(default_factory=dict)
    children: List["Step"] = field(default_factory=list)
    start: float = field(default_factory=time.perf_counter, repr=False)
    duration: float = 0.0

    def finalize(self) -> None:
        if self.duration == 0.0:
            self.duration = time.perf_counter() - self.start
        for child in self.children:
            child.finalize()

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dict (used for live updates)."""
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "content": self.content.copy(),
            "children": [child.to_dict() for child in self.children],
            "duration_ms": round(self.duration * 1000, 2),
        }


class TraceContext:
    """Context manager for a trace.

    Builds a nested Step tree, records LLM/tool/step data, and optionally
    streams live updates to the FlowTrace dashboard.
    """

    def __init__(self, name: str = "workflow", live: bool = False) -> None:
        self.name = name
        self.live = live
        self.trace_id = str(uuid.uuid4())
        self.start = datetime.now()
        self.root_step = Step(
            id=self.trace_id,
            name=name,
            kind="workflow",
        )
        self.success = True
        self._token: Optional[Token[Optional[TraceContext]]] = None
        self._current_step = self.root_step
        self._live_url: Optional[str] = None
        if live:
            base = os.getenv("FLOWTRACE_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
            self._live_url = f"{base}/api/traces/{self.trace_id}/live"

    def __enter__(self) -> TraceContext:
        self._token = _current_trace.set(self)
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        if exc_type is not None:
            self.success = False
            self._current_step.content.setdefault("status", "failed")
            self._current_step.content.setdefault("error", str(exc_val))
        self.root_step.finalize()
        if self.live and self._live_url:
            self._send_live_update()
        if self._token is not None:
            _current_trace.reset(self._token)

    def _send_live_update(self) -> None:
        if not self._live_url:
            return
        try:
            payload = {
                "name": self.name,
                "root_step": self.root_step.to_dict(),
                "start": self.start.isoformat(),
                "duration": round(self.root_step.duration, 3),
                "success": self.success,
                "total_tokens": self._count_tokens(),
            }
            httpx.post(self._live_url, json=payload, timeout=3.0)
        except Exception:
            # Live updates are best-effort and must never crash the agent.
            pass

    def _count_tokens(self) -> int:
        total = 0

        def visit(step: Step) -> None:
            nonlocal total
            if step.kind == "llm" and isinstance(step.content, dict):
                total += int(step.content.get("tokens", 0))
            for child in step.children:
                visit(child)

        visit(self.root_step)
        return total

    def record_llm_call(
        self,
        prompt: str = "",
        response: str = "",
        tokens: int = 0,
        latency: Optional[float] = None,
        **metadata: Any,
    ) -> None:
        content: Dict[str, Any] = {
            "prompt": prompt,
            "response": response,
            "tokens": tokens,
            "status": "completed",
        }
        if latency is not None:
            content["latency"] = latency
        content.update(metadata)
        step = Step(name="LLM Call", kind="llm", content=content)
        self._current_step.children.append(step)
        if self.live:
            self._send_live_update()

    def record_tool_call(
        self,
        name: str,
        args: Any = None,
        result: Any = None,
        **metadata: Any,
    ) -> None:
        content: Dict[str, Any] = {
            "input": args,
            "output": result,
            "status": "completed",
        }
        content.update(metadata)
        step = Step(name=name, kind="tool", content=content)
        self._current_step.children.append(step)
        if self.live:
            self._send_live_update()

    def record_step(
        self, name: str, kind: str = "step", **metadata: Any
    ) -> None:
        content: Dict[str, Any] = {"status": "completed", **metadata}
        step = Step(name=name, kind=kind, content=content)
        self._current_step.children.append(step)
        if self.live:
            self._send_live_update()

    @contextmanager
    def step(
        self, name: str, kind: str = "step", **metadata: Any
    ) -> Generator[None, None, None]:
        content: Dict[str, Any] = {"status": "started", **metadata}
        step = Step(
            name=name,
            kind=kind,
            content=content,
            start=time.perf_counter(),
        )
        self._current_step.children.append(step)
        previous = self._current_step
        self._current_step = step
        if self.live:
            self._send_live_update()
        try:
            yield
            step.content["status"] = "completed"
        except Exception as exc:
            self.success = False
            step.content["status"] = "failed"
            step.content["error"] = str(exc)
            raise
        finally:
            step.finalize()
            self._current_step = previous
            if self.live:
                self._send_live_update()


def trace(name: str = "workflow") -> Callable[[T], T]:
    """Decorator to automatically trace a function."""

    def decorator(func: T) -> T:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with TraceContext(name=name):
                return func(*args, **kwargs)

        return cast(T, wrapper)

    return decorator


def current_trace() -> TraceContext:
    """Return the active TraceContext (uses ContextVar for safety)."""
    ctx = _current_trace.get()
    if ctx is None:
        raise RuntimeError(
            "No active trace. Use `with TraceContext(...)` or the `@trace` decorator."
        )
    return ctx
