"""Official LangChain callback handler for FlowTrace.

Provides zero-code instrumentation for any LangChain LLM, tool, chain or agent.
Automatically builds a nested Step tree, records prompts/responses/tokens,
and persists the completed trace to SQLite.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from uuid import UUID

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError:
    BaseCallbackHandler = object  # type: ignore[assignment]


from .storage import SQLiteStore
from .tracer import Step, TraceContext


class FlowTraceHandler(BaseCallbackHandler):  # type: ignore[valid-type]
    """Callback handler that integrates LangChain with FlowTrace.

    Drop-in zero-code observability. Traces are built from LangChain
    run_id/parent_run_id events and automatically saved on root completion.

    Requires langchain-core (install via the ``flowtrace[langchain]`` extra).
    """

    def __init__(self, name: str = "LangChainWorkflow", live: bool = False) -> None:
        if BaseCallbackHandler is object:
            raise ImportError(
                "langchain-core is required for FlowTraceHandler. "
                'Install with: pip install "flowtrace[langchain]"'
            ) from None
        super().__init__()
        self.name = name
        self.live = live
        self.trace: Optional[TraceContext] = None
        self.steps: Dict[str, Step] = {}
        self.root_run_id: Optional[str] = None
        self.store: Optional[SQLiteStore] = None

    def _get_trace(self) -> TraceContext:
        if self.trace is None:
            self.trace = TraceContext(name=self.name, live=self.live)
            self.trace.__enter__()
            if self.store is None:
                self.store = SQLiteStore()
        return self.trace

    def _get_parent_step(self, parent_run_id: Optional[str]) -> Step:
        if parent_run_id and parent_run_id in self.steps:
            return self.steps[parent_run_id]
        trace = self.trace or self._get_trace()
        return trace.root_step

    def _parse_input(self, inp: Any) -> str:
        if not inp:
            return ""
        if isinstance(inp, list) and inp:
            item = inp[0]
            if hasattr(item, "content"):
                return str(item.content)
            if isinstance(item, dict) and "content" in item:
                return str(item["content"])
            return str(item)
        if isinstance(inp, dict):
            return str(inp.get("input", inp))
        return str(inp)

    def _parse_output(self, output: Any) -> str:
        if hasattr(output, "generations") and output.generations:
            gen_list = output.generations[0]
            if isinstance(gen_list, list) and gen_list:
                gen = gen_list[0]
                if hasattr(gen, "text"):
                    return str(gen.text)
            return str(gen_list)
        if hasattr(output, "content"):
            return str(output.content)
        if isinstance(output, dict):
            return str(output.get("output", output))
        return str(output)

    def _extract_token_count(self, response: Any) -> int:
        if hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
            usage = response.llm_output.get("token_usage") or response.llm_output.get("usage") or {}
            if isinstance(usage, dict):
                return int(usage.get("total_tokens", 0) or usage.get("total", 0))
        return 0

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str] | List[Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        trace = self._get_trace()
        rid = str(run_id)
        name = (
            serialized.get("name", "LLM Call")
            if isinstance(serialized, dict)
            else "LLM Call"
        )
        prompt = self._parse_input(prompts)
        content: Dict[str, Any] = {"prompt": prompt, "status": "started"}
        if "invocation_params" in kwargs:
            content["invocation_params"] = kwargs["invocation_params"]

        step = Step(
            id=rid,
            name=name,
            kind="llm",
            content=content,
            start=time.perf_counter(),
        )
        parent = self._get_parent_step(str(parent_run_id) if parent_run_id else None)
        parent.children.append(step)
        self.steps[rid] = step
        if self.root_run_id is None and parent_run_id is None:
            self.root_run_id = rid

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        step = self.steps.get(rid)
        if step is None:
            return
        step.content["response"] = self._parse_output(response)
        step.content["status"] = "completed"
        tokens = self._extract_token_count(response)
        if tokens > 0:
            step.content["tokens"] = tokens
        step.duration = time.perf_counter() - step.start
        self._maybe_finalize(rid)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        step = self.steps.get(rid)
        if step is None:
            return
        step.content.update({"error": str(error), "status": "failed"})
        step.duration = time.perf_counter() - step.start
        self._maybe_finalize(rid)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        trace = self._get_trace()
        rid = str(run_id)
        name = (
            serialized.get("name", "Tool")
            if isinstance(serialized, dict)
            else "Tool"
        )
        content: Dict[str, Any] = {"input": input_str, "status": "started"}
        step = Step(
            id=rid,
            name=name,
            kind="tool",
            content=content,
            start=time.perf_counter(),
        )
        parent = self._get_parent_step(str(parent_run_id) if parent_run_id else None)
        parent.children.append(step)
        self.steps[rid] = step
        if self.root_run_id is None and parent_run_id is None:
            self.root_run_id = rid

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        step = self.steps.get(rid)
        if step is None:
            return
        step.content["output"] = self._parse_output(output)
        step.content["status"] = "completed"
        step.duration = time.perf_counter() - step.start
        self._maybe_finalize(rid)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        step = self.steps.get(rid)
        if step is None:
            return
        step.content.update({"error": str(error), "status": "failed"})
        step.duration = time.perf_counter() - step.start
        self._maybe_finalize(rid)

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any] | str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        trace = self._get_trace()
        rid = str(run_id)
        name = (
            serialized.get("name", "Chain")
            if isinstance(serialized, dict)
            else "Chain"
        )
        content: Dict[str, Any] = {
            "inputs": self._parse_input(inputs),
            "status": "started",
        }
        step = Step(
            id=rid,
            name=name,
            kind="chain",
            content=content,
            start=time.perf_counter(),
        )
        parent = self._get_parent_step(str(parent_run_id) if parent_run_id else None)
        parent.children.append(step)
        self.steps[rid] = step
        if self.root_run_id is None and parent_run_id is None:
            self.root_run_id = rid

    def on_chain_end(
        self,
        outputs: Dict[str, Any] | str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        step = self.steps.get(rid)
        if step is None:
            return
        if isinstance(outputs, dict):
            step.content["outputs"] = outputs
        else:
            step.content["outputs"] = {"output": self._parse_output(outputs)}
        step.content["status"] = "completed"
        step.duration = time.perf_counter() - step.start
        self._maybe_finalize(rid)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        step = self.steps.get(rid)
        if step is None:
            return
        step.content.update({"error": str(error), "status": "failed"})
        step.duration = time.perf_counter() - step.start
        self._maybe_finalize(rid)

    def _maybe_finalize(self, run_id: str) -> None:
        if (
            self.root_run_id is not None
            and run_id == self.root_run_id
            and self.trace is not None
            and self.store is not None
        ):
            self.trace.__exit__(None, None, None)
            self.store.save_trace(self.trace)
            # reset handler for reuse / next trace
            self.trace = None
            self.root_run_id = None
            self.steps.clear()
            self.store = None
