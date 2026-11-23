"""Core tracing primitives for FlowTrace.

Provides TraceContext for manual instrumentation, the @trace decorator,
and the internal Step dataclass. Used by the LangChain and LlamaIndex handlers.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional


_current_trace: ContextVar[Optional[TraceContext]] = ContextVar(
    "current_trace", default=None
)


def current_trace() -> TraceContext:
    """Get the current active trace context."""
    trace = _current_trace.get()
    if trace is None:
        raise RuntimeError("No active trace. Use TraceContext or @trace.")
    return trace


@dataclass
class Step:
    """Step in the trace tree (dataclass used internally by TraceContext and handlers).

    Field order satisfies Python dataclass rules (non-defaults before defaults).
    """

    name: str
    kind: str = "step"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Dict[str, Any] = field(default_factory=dict)
    start: float = field(default_factory=time.perf_counter)
    duration: float = 0.0
    children: List[Step] = field(default_factory=list)


class TraceContext:
    """Trace context manager for manual instrumentation."""

    def __init__(self, name: str = "workflow", live: bool = False) -> None:
        self.id: str = str(uuid.uuid4())
        self.name: str = name
        self.live: bool = live
        self.start_time: datetime = datetime.now()
        self.start_perf: float = time.perf_counter()
        self.duration: float = 0.0
        self.success: bool = True
        self.total_tokens: int = 0
        self.root_step: Step = Step(
            name=name,
            kind="trace",
            start=self.start_perf,
            content={"status": "started"},
        )
        self._step_stack: List[Step] = [self.root_step]
        self._token: Optional[ContextVar.Token] = None

    def __enter__(self) -> TraceContext:
        self._token = _current_trace.set(self)
        return self

    def __exit__(
        self,
        exc_type: Optional[type] = None,
        exc_val: Optional[Exception] = None,
        exc_tb: Optional[Any] = None,
    ) -> None:
        self.duration = time.perf_counter() - self.start_perf
        if exc_type is not None:
            self.success = False
            self.root_step.content["status"] = "failed"
            self.root_step.content["error"] = str(exc_val)
        else:
            self.root_step.content["status"] = "completed"
        self.root_step.duration = self.duration
        if self.live:
            self._send_live_update()
        if self._token is not None:
            _current_trace.reset(self._token)

    def _send_live_update(self) -> None:
        """Broadcast current trace state to the dashboard via the live endpoint."""
        try:
            import httpx

            payload = {
                "id": self.id,
                "name": self.name,
                "root_step": self._step_to_dict(self.root_step),
                "start": self.start_time.isoformat(),
                "duration": round(self.duration, 4),
                "success": self.success,
                "total_tokens": self.total_tokens,
            }
            server_url = os.getenv("FLOWTRACE_SERVER_URL", "http://127.0.0.1:8080")
            httpx.post(
                f"{server_url}/api/traces/{self.id}/live",
                json=payload,
                timeout=1.5,
            )
        except Exception:
            pass  # live streaming is optional

    def _step_to_dict(self, step: Step) -> Dict[str, Any]:
        return {
            "id": step.id,
            "name": step.name,
            "kind": step.kind,
            "content": step.content,
            "start": step.start,
            "duration": round(step.duration, 4),
            "children": [self._step_to_dict(child) for child in step.children],
        }

    @contextmanager
    def step(
        self, name: str, kind: str = "step", metadata: Optional[Dict[str, Any]] = None
    ) -> Iterator[Step]:
        """Nested step context manager."""
        step = Step(
            name=name,
            kind=kind,
            content=metadata or {},
            start=time.perf_counter(),
        )
        self._step_stack[-1].children.append(step)
        self._step_stack.append(step)
        try:
            yield step
        except Exception as e:
            step.content["status"] = "failed"
            step.content["error"] = str(e)
            self.success = False
            raise
        else:
            step.content.setdefault("status", "completed")
        finally:
            step.duration = time.perf_counter() - step.start
            self._step_stack.pop()
            if self.live:
                self._send_live_update()

    def record_step(self, name: str, kind: str = "step", **kwargs: Any) -> None:
        """Record a completed step."""
        step = Step(
            name=name,
            kind=kind,
            content=kwargs,
            start=time.perf_counter(),
            duration=0.0,
        )
        self._step_stack[-1].children.append(step)
        if self.live:
            self._send_live_update()

    def record_llm_call(
        self,
        prompt: str,
        response: str = "",
        tokens: int = 0,
        latency: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Record LLM call details."""
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
            start=time.perf_counter() - latency,
            duration=latency,
        )
        self._step_stack[-1].children.append(step)
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
        """Record tool call details."""
        content = {
            "args": args,
            "result": result,
            "status": "completed",
            **kwargs,
        }
        step = Step(
            name=name,
            kind="tool",
            content=content,
            start=time.perf_counter(),
            duration=0.0,
        )
        self._step_stack[-1].children.append(step)
        if self.live:
            self._send_live_update()


def trace(name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for tracing functions."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tname = name or getattr(func, "__name__", "trace")
            with TraceContext(name=tname, live=False):
                return func(*args, **kwargs)
        return wrapper
    return decorator
