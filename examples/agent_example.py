"""Example research agent demonstrating FlowTrace usage.

Shows both the manual TraceContext/@trace API and the zero-code
LangChain callback handler (install with: pip install "flowtrace[langchain]").
"""

import json
import time
from typing import Any

from flowtrace import TraceContext, current_trace, trace


@trace("WebSearch")
def web_search(query: str) -> str:
    """Mock web search tool."""
    tc = current_trace()
    tc.record_tool_call(
        name="web_search",
        args={"query": query, "limit": 3},
        result={"results": ["Observability is critical for agents.", "FlowTrace gives local graphs."]},
    )
    time.sleep(0.12)  # simulate latency
    return "Found 2 relevant documents about LLM observability."


@trace("Summarizer")
def summarize(docs: str, question: str) -> str:
    """Mock LLM summarization step."""
    tc = current_trace()
    tc.record_llm_call(
        prompt=f"Question: {question}\nDocuments: {docs}\nSummarize:",
        response="FlowTrace provides zero-config tracing and interactive SVG graphs.",
        tokens=87,
        latency=0.45,
    )
    return "FlowTrace provides zero-config tracing and interactive SVG graphs."


def research_agent(query: str) -> str:
    """Multi-step research agent using manual instrumentation."""
    with TraceContext(name="ResearchAgent") as tc:
        tc.record_step("Start research", "setup", {"query": query})

        docs = web_search(query)
        answer = summarize(docs, query)

        tc.record_step("Final answer", "output", {"answer": answer})
        return answer


if __name__ == "__main__":
    print("Running manual FlowTrace example...")
    result = research_agent("LLM observability tools")
    print("Final answer:", result)
    print("\nRun `flowtrace serve` in another terminal to view the trace.")

    print("\n=== LangChain zero-code example (requires extra deps) ===")
    print("pip install \"flowtrace[langchain]\" langchain-core langchain-openai")
    print("""
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from flowtrace import FlowTraceHandler

handler = FlowTraceHandler(name="ResearchAgent")

llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[handler])

@tool
def web_search(query: str):
    ...

# agent = create_react_agent(llm=llm, tools=[web_search], ...)
# result = agent.invoke({"input": "LLM observability tools"})
# Trace is automatically saved to flowtrace.db
""")
