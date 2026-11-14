"""FastAPI router for FlowTrace.

Provides JSON endpoints consumed by the Swiss-minimal HTMX/Tailwind
dashboard and the HTTP ingestion path for non-Python agents.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Any, Dict, List, Union

from .models import Trace, TraceSummary
from .storage import SQLiteStore

router = APIRouter(prefix="/api", tags=["traces"])

# Singleton store (lazy)
_store: SQLiteStore | None = None


def get_store() -> SQLiteStore:
    """FastAPI dependency returning a shared SQLiteStore instance."""
    global _store
    if _store is None:
        _store = SQLiteStore()
    return _store


@router.get("/traces", response_model=List[TraceSummary])
def list_traces(
    limit: int = 50, store: SQLiteStore = Depends(get_store)
) -> List[TraceSummary]:
    """Return recent trace summaries (newest first)."""
    if limit < 1 or limit > 200:
        limit = 50
    return store.list_traces(limit=limit)


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
