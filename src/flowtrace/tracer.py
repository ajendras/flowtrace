"""Core tracing implementation for FlowTrace.

Provides TraceContext (context manager), @trace decorator, and
recording methods for LLM calls, tool calls, and generic steps.
Supports nested execution trees with automatic timing.
"""

from __future__ import annotations

import httpx
import os
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional


@dataclass
class Step:
    """A node in the execution trace tree.

    Production-ready step representation with automatic timing,
    nested children, and rich metadata for LLM calls and tools.
    """

    id: str
    name: str
    kind: str
    content: Dict[str, Any] = field(default_factory=dict)
    children: List["Step"] = field(default_factory=list)
    start: float = field(default_factory=time.perf_counter)
    duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert step (and children) to JSON-serializable dict."""
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "content": dict(self.content),
            "children": [c.to_dict() for c in self.children],
            "duration_ms": round(self.duration * 1000, 2),
        }


_trace_context: ContextVar[Optional[TraceContext]] = ContextVar(
    "trace_context", default=None
)


def current_trace() -> Optional[TraceContext]:
    """Return the active TraceContext for the current execution context."""
    return _trace_context.get()


def trace(
    name: str = "workflow", live: bool = False
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to trace a function with a TraceContext."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with TraceContext(name=name, live=live):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class TraceContext:
    """Main context manager for building nested trace trees.

    Can be used directly, with the @trace decorator, or by the framework
    handlers. Supports live streaming to the dashboard via HTTP POSTs.
    """

    def __init__(self, name: str = "workflow", live: bool = False) -> None:
        self.name = name
        self.live = live
        self.trace_id = str(uuid.uuid4())
        self.root_step = Step(
            id=self.trace_id,
            name=name,
            kind="trace",
            start=time.perf_counter(),
        )
        self.current_step: List[Step] = [self.root_step]
        self.start = datetime.now()
        self.success = True
        self.total_tokens = 0
        self._token: Optional[object] = None

    def __enter__(self) -> TraceContext:
        self._token = _trace_context.set(self)
        if self.live:
            self._send_live_update()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.root_step.duration = time.perf_counter() - self.root_step.start
        if exc_type is not None:
            self.success = False
            if self.current_step:
                current = self.current_step[-1]
                current.content["status"] = "failed"
                current.content["error"] = str(exc_val)
        if self.live:
            self._send_live_update()
        if self._token is not None:
            _trace_context.reset(self._token)
        self._token = None

    def get_root_step(self) -> Step:
        """Return the root step for model conversion."""
        return self.root_step

    def record_llm_call(
        self,
        prompt: str,
        response: str,
        tokens: int = 0,
        latency: Optional[float] = None,
    ) -> None:
        """Record an LLM call as a step in the trace."""
        content: Dict[str, Any] = {
            "prompt": prompt,
            "response": response,
            "tokens": tokens,
            "status": "completed",
        }
        if latency is not None:
            content["latency"] = latency
        self.record_step("LLM Call", kind="llm", content=content)
        self.total_tokens += tokens

    def record_tool_call(
        self,
        name: str,
        args: Dict[str, Any],
        result: Any,
        latency: Optional[float] = None,
    ) -> None:
        """Record a tool invocation as a step in the trace."""
        content: Dict[str, Any] = {
            "args": args,
            "result": result,
            "status": "completed",
        }
        if latency is not None:
            content["latency"] = latency
        self.record_step(name, kind="tool", content=content)

    def record_step(
        self,
        name: str,
        kind: str = "step",
        content: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a generic step (leaf node)."""
        content = content or {}
        step = Step(
            id=str(uuid.uuid4()),
            name=name,
            kind=kind,
            content=dict(content),
            start=time.perf_counter(),
            duration=0.0,
        )
        self.current_step[-1].children.append(step)
        if self.live:
            self._send_live_update()

    @contextmanager
    def step(
        self,
        name: str,
        kind: str = "step",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Step]:
        """Context manager for a nested step. Supports metadata passed into content."""
        content: Dict[str, Any] = dict(metadata) if metadata else {}
        step = Step(
            id=str(uuid.uuid4()),
            name=name,
            kind=kind,
            content=content,
            start=time.perf_counter(),
        )
        parent = self.current_step[-1]
        parent.children.append(step)
        self.current_step.append(step)
        if self.live:
            self._send_live_update()
        try:
            yield step
        except Exception as exc:
            step.content["status"] = "failed"
            step.content["error"] = str(exc)
            self.success = False
            raise
        finally:
            step.duration = time.perf_counter() - step.start
            self.current_step.pop()
            if self.live:
                self._send_live_update()

    def _send_live_update(self) -> None:
        """Send current trace state to the live update endpoint."""
        payload = {
            "name": self.name,
            "root_step": self.root_step.to_dict(),
            "start": self.start.isoformat(),
            "duration": self.root_step.duration,
            "success": self.success,
            "total_tokens": self.total_tokens,
        }
        self._live_post(payload)

    def _live_post(self, payload: Dict[str, Any]) -> None:
        """Best-effort HTTP POST to the live endpoint."""
        base_url = os.getenv("FLOWTRACE_SERVER_URL", "http://127.0.0.1:8080")
        url = f"{base_url}/api/traces/{self.trace_id}/live"
        try:
            httpx.post(url, json=payload, timeout=0.5)
        except Exception:
            # Live streaming is best effort; do not fail the trace on network issues.
            pass
