"""Core tracing implementation for FlowTrace.

Provides TraceContext (context manager), @trace decorator, and
recording methods for LLM calls, tool calls, and generic steps.
Supports nested execution trees with automatic timing.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Dict, List, Optional, Union


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
        """Convert step (and children) to JSON-serializable dict.

        Used by the API layer and frontend SVG renderer.
        Durations are reported in milliseconds.
        """
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "content": self.content,
            "children": [child.to_dict() for child in self.children],
            "duration_ms": round(self.duration * 1000, 2),
        }


# Thread-safe and async-safe current trace using contextvars.
_current_trace_var: ContextVar[Optional[TraceContext]] = ContextVar(
    "flowtrace_current", default=None
)


def current_trace() -> Optional[TraceContext]:
    """Return the active TraceContext for the current execution context.

    Returns None if no trace is active. Used by decorators, manual
    instrumentation, and library code.
    """
    return _current_trace_var.get()


class TraceContext:
    """Main tracing context and step recorder.

    Used as a context manager to start a trace and automatically
    finalise timing on exit. Maintains a stack for nested steps.
    All public recording methods are thread-safe within the same
    contextvar scope.
    """

    def __init__(self, name: str = "workflow") -> None:
        """Create a new trace with a root step.

        Args:
            name: Human-readable name for the root trace step.
        """
        self.trace_id: str = str(uuid.uuid4())
        self.name: str = name
        self.start_time: float = time.perf_counter()

        self.root_step = Step(
            id=self.trace_id,
            name=name,
            kind="trace",
            start=self.start_time,
        )

        self.step_stack: List[Step] = [self.root_step]
        self.token: Optional[object] = None

    def __enter__(self) -> TraceContext:
        """Enter the context, register as current trace."""
        self.token = _current_trace_var.set(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context, finalise root duration and error state."""
        end_time = time.perf_counter()
        self.root_step.duration = end_time - self.start_time

        if exc_type is not None:
            error_data: Dict[str, Any] = {
                "error_type": exc_type.__name__,
                "error_message": str(exc_val),
                "status": "failed",
            }
            self.record_step("Execution Error", "error", error_data)

        if self.token is not None:
            _current_trace_var.reset(self.token)

    @contextmanager
    def step(
        self,
        name: str,
        kind: str = "step",
        data: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for a nested timed step.

        Automatically records start time on entry and duration on exit.
        Supports arbitrary metadata via the data dict. Used internally
        by the decorator and record_* methods.
        """
        if data is None:
            data = {}

        step = Step(
            id=str(uuid.uuid4()),
            name=name,
            kind=kind,
            content=data.copy(),  # defensive copy
            start=time.perf_counter(),
        )

        parent = self.step_stack[-1]
        parent.children.append(step)
        self.step_stack.append(step)

        try:
            yield step
        except Exception as exc:  # pylint: disable=broad-except
            step.content.setdefault("error", str(exc))
            step.content.setdefault("status", "failed")
            raise
        finally:
            end = time.perf_counter()
            step.duration = end - step.start
            self.step_stack.pop()

    def record_step(
        self,
        name: str,
        kind: str = "step",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an instantaneous completed step.

        Useful for events that do not need their own nested children.
        The step is added to the current parent in the trace tree.

        Args:
            name: The name of the step/event.
            kind: The category of step (default: "step"). Common values:
                "llm", "tool", "error", "setup", "phase".
            data: Optional dict of metadata to attach.
        """
        if data is None:
            data = {}
        # Use the context manager with an empty body for consistent timing logic.
        with self.step(name, kind, data):
            pass

    def record_llm_call(
        self,
        prompt: str,
        response: str,
        tokens: int,
        latency: Optional[float] = None,
    ) -> None:
        """Record an LLM invocation with prompt, completion and usage stats.

        Args:
            prompt: The raw prompt sent to the model.
            response: The model output.
            tokens: Total tokens used (input+output).
            latency: Optional wall-clock latency in seconds.
        """
        if latency is None:
            latency = 0.0

        data: Dict[str, Any] = {
            "prompt": prompt,
            "response": response,
            "tokens": tokens,
            "latency_ms": round(latency * 1000, 2),
        }
        self.record_step("LLM Call", "llm", data)

    def record_tool_call(
        self, name: str, args: Dict[str, Any], result: Any
    ) -> None:
        """Record a tool invocation with arguments and return value.

        Args:
            name: Tool name (e.g. 'search', 'calculator').
            args: Dictionary of arguments passed to the tool.
            result: Tool output (will be stringified if complex).
        """
        output = (
            result
            if isinstance(result, (str, int, float, bool, dict, list))
            else str(result)
        )
        data: Dict[str, Any] = {
            "name": name,
            "arguments": args,
            "output": output,
        }
        self.record_step(name, "tool", data)

    def get_root_step(self) -> Step:
        """Return the root step of the trace tree.

        Used by storage, API, and graph rendering layers.
        """
        return self.root_step

    def to_dict(self) -> Dict[str, Any]:
        """Convenience method to export the entire trace as a dict.

        Includes trace metadata and the full nested step tree.
        """
        root = self.root_step.to_dict()
        root["trace_id"] = self.trace_id
        root["total_duration_ms"] = round(
            (time.perf_counter() - self.start_time) * 1000, 2
        )
        return root


def trace(
    name: Optional[Union[str, Callable[..., Any]]] = None, kind: str = "step"
) -> Callable[..., Any]:
    """Decorator to automatically trace a function as a named step.

    Supports two usage patterns:
        @trace
        def my_agent(...):
            ...

        @trace("Research Phase")
        def research(...):
            ...

    If no active TraceContext exists, a new top-level trace is created.
    Otherwise the function becomes a nested step. LLM and tool calls
    made inside the function will be captured if they use current_trace().
    """

    def _decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        step_name = name if isinstance(name, str) else func.__name__
        step_kind = kind

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tc = current_trace()

            if tc is None:
                # Top-level trace
                with TraceContext(name=step_name) as tc:
                    with tc.step(step_name, step_kind):
                        return func(*args, **kwargs)
            else:
                # Nested step inside existing trace
                with tc.step(step_name, step_kind):
                    return func(*args, **kwargs)

        return wrapper

    # Support bare @trace (name is the function) and @trace("name")
    if callable(name):
        return _decorate(name)

    return _decorate
