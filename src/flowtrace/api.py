"""FastAPI router for FlowTrace.

Provides JSON endpoints consumed by the Swiss-minimal HTMX/Tailwind
dashboard and the HTTP ingestion path for non-Python agents.
"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from typing import Any, Dict, List, Optional, Union

from .models import Trace, TraceSummary, TraceComparison
from .storage import SQLiteStore


class ConnectionManager:
    """Manages WebSocket connections for live trace updates per trace_id."""

    def __init__(self) -> None:
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, trace_id: str) -> None:
        await websocket.accept()
        if trace_id not in self.active_connections:
            self.active_connections[trace_id] = []
        self.active_connections[trace_id].append(websocket)

    def disconnect(self, websocket: WebSocket, trace_id: str) -> None:
        if trace_id in self.active_connections:
            if websocket in self.active_connections[trace_id]:
                self.active_connections[trace_id].remove(websocket)
            if not self.active_connections[trace_id]:
                del self.active_connections[trace_id]

    async def broadcast(self, trace_id: str, message: Dict[str, Any]) -> None:
        if trace_id not in self.active_connections:
            return
        for connection in list(self.active_connections[trace_id]):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection, trace_id)


router = APIRouter(prefix="/api", tags=["traces"])

# Singleton store (lazy)
_store: SQLiteStore | None = None
manager = ConnectionManager()


def get_store() -> SQLiteStore:
    """FastAPI dependency returning a shared SQLiteStore instance."""
    global _store
    if _store is None:
        _store = SQLiteStore()
    return _store


def _count_steps(step: Any) -> int:
    """Recursively count steps (including the current one)."""
    if step is None:
        return 0
    count = 1
    for child in getattr(step, "children", []):
        count += _count_steps(child)
    return count


@router.get("/traces", response_model=List[TraceSummary])
def list_traces(
    limit: int = 50,
    q: Optional[str] = None,
    store: SQLiteStore = Depends(get_store)
) -> List[TraceSummary]:
    """Return recent trace summaries (newest first).

    The optional ``q`` parameter performs a full-text search over trace names,
    prompt texts, LLM responses, tool call arguments/outputs and other content.
    """
    if limit < 1 or limit > 200:
        limit = 50
    return store.list_traces(limit=limit, search=q)


@router.post("/traces", status_code=status.HTTP_201_CREATED)
def ingest_trace(
    payload: Union[Dict[str, Any], Trace], store: SQLiteStore = Depends(get_store)
) -> Dict[str, str]:
    """Ingest a trace from Python code or remote HTTP POST.

    Accepts raw dict (common for external agents) or a validated Trace.
    """
    if isinstance(payload, dict):
        try:
            trace = Trace.model_validate(payload)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid trace payload: {exc}",
            ) from exc
    else:
        trace = payload

    trace_id = store.save_trace(trace)
    return {"id": trace_id, "status": "saved"}


@router.get("/traces/{trace_id}", response_model=Trace)
def get_trace(
    trace_id: str, store: SQLiteStore = Depends(get_store)
) -> Trace:
    """Return the complete trace (including full nested step tree)."""
    trace = store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(
            status_code=404,
            detail=f"Trace with id '{trace_id}' not found",
        )
    return trace


@router.get("/traces/{trace_id}/graph")
def get_trace_graph(
    trace_id: str, store: SQLiteStore = Depends(get_store)
) -> Any:
    """Return only the root Step tree for the interactive SVG renderer.

    Smaller payload; the frontend only needs the graph structure.
    """
    trace = store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(
            status_code=404,
            detail=f"Trace with id '{trace_id}' not found",
        )
    return trace.root_step


@router.get("/compare/{trace_id1}/{trace_id2}", response_model=TraceComparison)
def compare_traces(
    trace_id1: str,
    trace_id2: str,
    store: SQLiteStore = Depends(get_store),
) -> TraceComparison:
    """Return two complete traces with computed differences for the
    side-by-side comparison view.
    """
    trace1 = store.get_trace(trace_id1)
    trace2 = store.get_trace(trace_id2)
    if trace1 is None or trace2 is None:
        raise HTTPException(
            status_code=404,
            detail="One or both traces not found",
        )

    diff = {
        "duration_diff": round(abs(trace1.duration - trace2.duration), 3),
        "token_diff": abs(trace1.total_tokens - trace2.total_tokens),
        "step_count_diff": abs(
            _count_steps(trace1.root_step) - _count_steps(trace2.root_step)
        ),
        "success_match": trace1.success == trace2.success,
    }

    return TraceComparison(
        trace1=trace1,
        trace2=trace2,
        differences=diff,
    )


@router.get("/traces/{trace_id}/export")
def export_trace_endpoint(
    trace_id: str,
    format: str = "otel",
    store: SQLiteStore = Depends(get_store),
) -> Any:
    """Export trace in OpenTelemetry or LangSmith compatible format.

    ``?format=otel`` returns OTLP JSON; ``?format=langsmith`` returns a list of runs.
    """
    trace = store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(
            status_code=404,
            detail=f"Trace with id '{trace_id}' not found",
        )

    if format.lower() in ("otel", "opentelemetry", "otlp"):
        return trace.to_opentelemetry()
    if format.lower() in ("langsmith",):
        return trace.to_langsmith()

    raise HTTPException(
        status_code=400,
        detail="format must be 'otel' or 'langsmith'",
    )


@router.websocket("/ws/traces/{trace_id}")
async def websocket_trace_stream(websocket: WebSocket, trace_id: str):
    """WebSocket endpoint for live trace streaming during runs.

    Frontend connects here to receive real-time step and duration updates.
    """
    await manager.connect(websocket, trace_id)
    try:
        store = get_store()
        trace = store.get_trace(trace_id)
        if trace is not None:
            await websocket.send_json(
                {
                    "type": "trace_update",
                    "trace": trace.model_dump(mode="json"),
                }
            )
        while True:
            # Keep connection alive; client may send pings. On disconnect the
            # exception is raised and we clean up.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, trace_id)
    except Exception:
        manager.disconnect(websocket, trace_id)


@router.post("/traces/{trace_id}/live", status_code=status.HTTP_202_ACCEPTED)
async def live_trace_update(
    trace_id: str,
    payload: Dict[str, Any],
    store: SQLiteStore = Depends(get_store),
) -> Dict[str, str]:
    """Accept live updates from a TraceContext(live=True) and broadcast
    via WebSocket to all connected dashboard clients.
    """
    try:
        trace_dict: Dict[str, Any] = {
            "id": trace_id,
            "name": payload.get("name", "workflow"),
            "root_step": payload.get("root_step"),
            "start": payload.get("start"),
            "duration": payload.get("duration", 0.0),
            "success": payload.get("success", True),
            "total_tokens": payload.get("total_tokens", 0),
        }
        trace = Trace.model_validate(trace_dict)
        store.save_trace(trace)
    except Exception:
        # Still broadcast even if a partial trace cannot be saved yet
        pass

    await manager.broadcast(
        trace_id,
        {
            "type": "trace_update",
            "trace_id": trace_id,
            "root_step": payload.get("root_step"),
            "duration": payload.get("duration", 0.0),
            "success": payload.get("success", True),
        },
    )
    return {"status": "broadcasted"}
