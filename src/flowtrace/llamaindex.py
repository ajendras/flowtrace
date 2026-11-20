"""Official LlamaIndex callback handler for FlowTrace.

Provides zero-code instrumentation for any LlamaIndex query engine,
agent, LLM, retriever or embedding. Automatically builds a nested
Step tree from CBEventType events and persists the completed trace.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

try:
    from llama_index.core.callbacks.base import BaseCallbackHandler
    from llama_index.core.callbacks.schema import CBEventType, EventPayload
except ImportError:
    BaseCallbackHandler = object  # type: ignore[assignment]
    CBEventType = object  # type: ignore[assignment]
    EventPayload = object  # type: ignore[assignment]


from .storage import SQLiteStore
from .tracer import Step, TraceContext


class LlamaIndexHandler(BaseCallbackHandler):  # type: ignore[valid-type]
    """Callback handler that integrates LlamaIndex with FlowTrace.

    Drop-in zero-code observability. Traces are built from LlamaIndex
    CBEventType events (LLM, QUERY, RETRIEVE, EMBEDDING, TOOL, AGENT, ...)
    and automatically saved on root completion.

    Requires llama-index-core (install via the ``flowtrace[llamaindex]`` extra).
    """

    def __init__(self, name: str = "LlamaIndexWorkflow", live: bool = False) -> None:
        if BaseCallbackHandler is object:
            raise ImportError(
                "llama-index-core is required for LlamaIndexHandler. "
                'Install with: pip install "flowtrace[llamaindex]"'
            ) from None
        super().__init__()
        self.name = name
        self.live = live
        self.trace: Optional[TraceContext] = None
        self.steps: Dict[str, Step] = {}
        self.root_event_id: Optional[str] = None
        self.store: Optional[SQLiteStore] = None

    def _get_trace(self) -> TraceContext:
        if self.trace is None:
            self.trace = TraceContext(name=self.name, live=self.live)
            self.trace.__enter__()
            self.store = SQLiteStore()
        return self.trace

    def _get_parent_step(self, parent_id: Optional[str]) -> Step:
        if parent_id and parent_id in self.steps:
            return self.steps[parent_id]
        trace = self.trace or self._get_trace()
        return trace.root_step

    def _event_type_to_kind(self, event_type: Any) -> str:
        if not hasattr(event_type, "value"):
            et = str(event_type).lower()
        else:
            et = event_type.value.lower()
        if "llm" in et:
            return "llm"
        if any(k in et for k in ("tool", "function")):
            return "tool"
        if "retriev" in et:
            return "retrieval"
        if "embed" in et:
            return "embedding"
        if any(k in et for k in ("query", "chat")):
            return "query"
        if "agent" in et:
            return "agent"
        return "step"

    def _parse_payload(
        self, payload: Optional[Dict[str, Any]], event_type: Any, is_end: bool = False
    ) -> Dict[str, Any]:
        if not payload:
            return {}
        content: Dict[str, Any] = {}

        # Common payload keys used by LlamaIndex
        if "query_str" in payload:
            content["query"] = str(payload["query_str"])
        if (
            hasattr(EventPayload, "QUERY_STR")
            and EventPayload.QUERY_STR in payload  # type: ignore[attr-defined]
        ):
            content["query"] = str(payload[EventPayload.QUERY_STR])  # type: ignore[attr-defined]

        if "prompt" in payload:
            content["prompt"] = str(payload["prompt"])
        if (
            hasattr(EventPayload, "PROMPT")
            and EventPayload.PROMPT in payload  # type: ignore[attr-defined]
        ):
            content["prompt"] = str(payload[EventPayload.PROMPT])  # type: ignore[attr-defined]

        if is_end:
            for key in ("response", "completion", "output"):
                if key in payload:
                    content["response"] = str(payload[key])
                    break
            if (
                hasattr(EventPayload, "RESPONSE")
                and EventPayload.RESPONSE in payload  # type: ignore[attr-defined]
            ):
                content["response"] = str(payload[EventPayload.RESPONSE])  # type: ignore[attr-defined]
            if (
                hasattr(EventPayload, "COMPLETION")
                and EventPayload.COMPLETION in payload  # type: ignore[attr-defined]
            ):
                content["response"] = str(payload[EventPayload.COMPLETION])  # type: ignore[attr-defined]

        if "tool" in payload or (
            hasattr(EventPayload, "TOOL")
            and EventPayload.TOOL in payload  # type: ignore[attr-defined]
        ):
            tool = payload.get("tool")
            if not tool and hasattr(EventPayload, "TOOL"):
                tool = payload.get(EventPayload.TOOL)  # type: ignore[attr-defined]
            if tool:
                content["tool"] = str(tool)

        # Token counts (various possible keys)
        token_keys = ["total_tokens", "token_count", "prompt_tokens"]
        if hasattr(EventPayload, "TOTAL_TOKENS"):
            token_keys.append(EventPayload.TOTAL_TOKENS)  # type: ignore[attr-defined]
        for tk in token_keys:
            if tk in payload and payload.get(tk):
                content["tokens"] = int(payload[tk])
                break

        if is_end and (
            (
                hasattr(EventPayload, "EXCEPTION")
                and EventPayload.EXCEPTION in payload  # type: ignore[attr-defined]
            )
            or "error" in payload
            or "exception" in payload
        ):
            exc = payload.get("error") or payload.get("exception")
            if (
                hasattr(EventPayload, "EXCEPTION")
                and EventPayload.EXCEPTION in payload  # type: ignore[attr-defined]
            ):
                exc = payload.get(EventPayload.EXCEPTION)  # type: ignore[attr-defined]
            if exc:
                content["error"] = str(exc)
                content["status"] = "failed"

        return content

    def on_event_start(
        self,
        event_type: Any,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> str:
        trace = self._get_trace()
        rid = event_id or str(uuid.uuid4())
        name = (
            getattr(event_type, "value", None)
            or getattr(event_type, "name", None)
            or str(event_type)
        )
        kind = self._event_type_to_kind(event_type)
        content = {"status": "started"}
        content.update(self._parse_payload(payload, event_type))

        step = Step(
            id=rid,
            name=name,
            kind=kind,
            content=content,
            start=time.perf_counter(),
        )
        parent_id = kwargs.get("parent_id")
        parent = self._get_parent_step(parent_id)
        parent.children.append(step)
        self.steps[rid] = step
        if self.root_event_id is None:
            self.root_event_id = rid
        return rid

    def on_event_end(
        self,
        event_type: Any,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        rid = event_id
        step = self.steps.get(rid)
        if step is None:
            return
        parsed = self._parse_payload(payload, event_type, is_end=True)
        step.content.update(parsed)
        if "error" not in step.content:
            step.content["status"] = "completed"
        step.duration = time.perf_counter() - step.start
        self._maybe_finalize(rid)

    def _maybe_finalize(self, event_id: str) -> None:
        if (
            self.root_event_id is not None
            and event_id == self.root_event_id
            and self.trace is not None
            and self.store is not None
        ):
            self.trace.__exit__(None, None, None)
            self.store.save_trace(self.trace)
            # reset handler for reuse / next trace
            self.trace = None
            self.root_event_id = None
            self.steps.clear()
            self.store = None

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Optional hook for top-level trace start (no-op)."""
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Finalize root trace if no specific root event was tracked."""
        if self.trace is not None and self.root_event_id is None:
            self.trace.__exit__(None, None, None)
            if self.store is not None:
                self.store.save_trace(self.trace)
            self.trace = None
            self.steps.clear()
            self.store = None
