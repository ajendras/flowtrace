"""FlowTrace - Visualizes and debugs multi-step LLM agent workflows.

Zero-config tracing for Python agents with interactive SVG graphs,
prompt inspection, tool call browser, and local SQLite persistence.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Optional, Union

from .tracer import TraceContext, current_trace, trace

__version__ = "0.1.0"
__all__ = ["trace", "TraceContext", "current_trace", "run_server", "FlowTraceHandler"]


def run_server(port: int = 8080, host: str = "127.0.0.1") -> None:
    """Start the FlowTrace dashboard server.

    Args:
        port: Port to listen on (default 8080).
        host: Host address to bind to (default 127.0.0.1).

    The server provides the Swiss-minimal UI, API endpoints,
    and serves static assets.
    """
    # Defer imports until runtime so that the core tracer can be
    # used without pulling in the full web stack.
    from .server import create_app
    import uvicorn

    app = create_app()
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )


def __getattr__(name: str) -> Any:
    """Lazy-load FlowTraceHandler and the langchain submodule.

    This ensures langchain-core is not imported unless the handler
    or langchain integration is explicitly used. Also supports test
    patching of flowtrace.langchain.
    """
    if name == "FlowTraceHandler":
        from .langchain import FlowTraceHandler as _Handler

        globals()["FlowTraceHandler"] = _Handler
        return _Handler
    if name == "langchain":
        from . import langchain as _langchain

        globals()["langchain"] = _langchain
        return _langchain
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
