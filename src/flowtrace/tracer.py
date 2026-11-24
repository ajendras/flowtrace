"""Core tracing primitives for FlowTrace.

Zero-config decorator, context manager, and nested step builder with
contextvars for safe concurrent use. Supports live streaming to the
dashboard.
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

current: ContextVar[Optional["TraceContext"]] = ContextVar(
    "current_trace", default=None
)


def current_trace() -> Optional[TraceContext]:
    """Get the trace context for the current execution context."""
    return current.get()


T = TypeVar("T", bound=Callable[..., Any])


def trace(name_or_func: Union[str, T] = "workflow") -> Union[Callable[[T], T], T]:
    """Decorator or decorator factory for tracing functions.

    Supports both ``@trace`` and ``@trace("CustomName")``.
    """
    def _make_wrapper(func: T, name: str) -> T:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with TraceContext(name=name, live=False) as tc:
                return func(*args, **kwargs)
        return cast(T, wrapper)

    if callable(name_or_func):
        func = cast(T, name_or_func)
        name = getattr(func, "__name__", "workflow")
        return _make_wrapper(func, name)

    name = name_or_func
    def decorator(func: T) -> T:
        return _make_wrapper(func, name)
    return decorator


@dataclass
class Step:
    """A step in the trace tree.

    Field order satisfies the dataclass rule that non-default fields must
    precede fields with defaults (this was the root cause of the CI failure).
    """
    name: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    kind: str = "step"
    content: Dict[str, Any] = field(default_factory=dict)
    start: float = field(default_factory=time.perf_counter)
    duration: float = 0.0
    children: List["Step"] = field(default_factory=list)
    annotations: List[Dict[str, Any]] = field(default_factory=list)

    def annotate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an annotation to this step (persisted and shown in UI)."""
        self.annotations.append(
            {
                "text": text,
                "timestamp": time.time(),
                "metadata": metadata or {},
            }
        )

    def as_dict(self) -> Dict[str, Any]:
        """Convert tree to dict for live updates and serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "content": dict(self.content),
            "duration": round(self.duration, 6),
            "children": [c.as_dict() for c in self.children],
            "annotations": list(self.annotations),
        }


class TraceContext:
    """Main context for building a trace.

    Manages nested steps, timing, live updates, recording helpers, and
    persistence via SQLite.
    """

    def __init__(self, name: str = "workflow", live: bool = False) -> None:
        self.id = str(uuid.uuid4())
        self.name = name
        self.start = time.perf_counter()
        self.duration = 0.0
        self.success = True
        self.total_tokens = 0
        self.root_step = Step(
            name=name,
            kind="root",
            id=self.id,
            start=self.start,
        )
        self.live = live
        self._token: Optional[object] = None
        self._current_step: Optional[Step] = self.root_step
        self._store: Optional[Any] = None

    def __enter__(self) -> TraceContext:
        self._token = current.set(self)
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self.duration = time.perf_counter() - self.start
        if exc_type is not None:
            self.success = False
            self.root_step.content["error"] = str(exc_val)
            self.root_step.content["status"] = "failed"
        if self.live:
            self._broadcast_live()
        self._finalize()
        if self._token is not None:
            current.reset(self._token)

    def _finalize(self) -> None:
        """Persist the completed trace."""
        from .models import Trace
        from .storage import SQLiteStore

        if self._store is None:
            self._store = SQLiteStore()
        trace_model = Trace.from_tracer_context(self)
        self._store.save_trace(trace_model)

    def _broadcast_live(self) -> None:
        """Best-effort live update to the dashboard via HTTP."""
        import os

        base_url = os.getenv("FLOWTRACE_SERVER_URL", "http://127.0.0.1:8080")
        try:
            import httpx

            payload = {
                "name": self.name,
                "root_step": self.root_step.as_dict(),
                "duration": self.duration,
                "success": self.success,
                "total_tokens": self.total_tokens,
            }
            httpx.post(
                f"{base_url}/api/traces/{self.id}/live",
                json=payload,
                timeout=2.0,
            )
        except Exception:
            pass  # live updates are best-effort

    def step(
        self,
        name: str,
        kind: str = "step",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StepContextManager:
        """Context manager for a nested step (supports ``as step``)."""
        content = dict(metadata) if metadata else {}
        step = Step(name=name, kind=kind, content=content, start=time.perf_counter())
        if self._current_step is not None:
            self._current_step.children.append(step)
        else:
            self.root_step.children.append(step)
        return StepContextManager(self, step)

    def record_step(self, name: str, kind: str = "step") -> Step:
        """Record a step and return it (for later annotation)."""
        step = Step(name=name, kind=kind, start=time.perf_counter())
        if self._current_step is not None:
            self._current_step.children.append(step)
        else:
            self.root_step.children.append(step)
        return step

    def record_llm_call(
        self,
        prompt: str,
        response: str = "",
        tokens: int = 0,
        latency: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Record an LLM call step."""
        content: Dict[str, Any] = {
            "prompt": prompt,
            "response": response,
            "status": "completed",
            "tokens": tokens,
        }
        content.update(kwargs)
        if tokens > 0:
            self.total_tokens += tokens
        step = Step(
            name="LLM Call",
            kind="llm",
            content=content,
            start=time.perf_counter(),
        )
        if latency is not None:
            step.duration = latency
        if self._current_step is not None:
            self._current_step.children.append(step)
        else:
            self.root_step.children.append(step)

    def record_tool_call(
        self,
        name: str,
        args: Dict[str, Any],
        result: Any,
        **kwargs: Any,
    ) -> None:
        """Record a tool call step."""
        content: Dict[str, Any] = {
            "args": args,
            "result": result,
            "status": "completed",
        }
        content.update(kwargs)
        step = Step(
            name=name,
            kind="tool",
            content=content,
            start=time.perf_counter(),
        )
        if self._current_step is not None:
            self._current_step.children.append(step)
        else:
            self.root_step.children.append(step)

    def annotate(
        self,
        text: str,
        step_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add annotation to the root or a specific step by id."""
        if not step_id:
            self.root_step.annotate(text, metadata)
            return

        def find(s: Step) -> bool:
            if s.id == step_id:
                s.annotate(text, metadata)
                return True
            for child in s.children:
                if find(child):
                    return True
            return False

        find(self.root_step)


class StepContextManager:
    """Context manager returned by ``TraceContext.step()``."""

    def __init__(self, tc: TraceContext, step: Step) -> None:
        self.tc = tc
        self.step = step
        self.parent: Optional[Step] = tc._current_step

    def __enter__(self) -> Step:
        self.tc._current_step = self.step
        return self.step

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.step.duration = time.perf_counter() - self.step.start
        if exc_type is not None:
            self.step.content["status"] = "failed"
            self.step.content["error"] = str(exc_val)
            self.tc.success = False
        self.tc._current_step = self.parent
        if self.tc.live:
            self.tc._broadcast_live()
