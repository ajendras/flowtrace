"""Command-line interface for FlowTrace.

Provides ``flowtrace serve``, ``list``, ``view``, and ``clear`` commands
with clear output and input validation. Uses the shared SQLiteStore for
all persistence operations.
"""

from __future__ import annotations

import sys
from typing import Optional

import typer

from . import run_server
from .storage import SQLiteStore

app = typer.Typer(
    name="flowtrace",
    help="FlowTrace — Visualizes and debugs multi-step LLM agent workflows",
    add_completion=False,
)


@app.command()
def serve(
    port: int = typer.Option(8080, "--port", "-p", help="Port to listen on", min=1, max=65535),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host address to bind"),
) -> None:
    """Start the FlowTrace Swiss-minimal dashboard server."""
    typer.echo(f"Starting FlowTrace at http://{host}:{port}")
    typer.echo("Press Ctrl+C to stop.")
    run_server(port=port, host=host)


@app.command()
def list(limit: int = typer.Option(20, "--limit", "-n", help="Number of traces to show", min=1, max=200)) -> None:
    """List recent traces (newest first)."""
    store = SQLiteStore()
    traces = store.list_traces(limit=limit)

    if not traces:
        typer.echo("No traces recorded yet. Run an instrumented agent or POST to /api/traces.")
        return

    # Simple aligned text table (no extra dependencies)
    headers = ["ID", "NAME", "START", "DURATION", "TOKENS", "STATUS"]
    widths = [10, 18, 20, 10, 8, 8]

    def pad(text: str, width: int, right: bool = False) -> str:
        if right:
            return text.rjust(width)
        return text.ljust(width)

    header_line = "  ".join(pad(h, w) for h, w in zip(headers, widths))
    typer.echo(header_line)
    typer.echo("-" * len(header_line))

    for t in traces:
        tid = t.id[:8]
        name = t.name[:16] + ("..." if len(t.name) > 16 else "")
        start_str = t.start.strftime("%Y-%m-%d %H:%M")
        duration = f"{t.duration_ms:.0f}ms"
        tokens = str(t.total_tokens)
        status = "OK" if t.success else "ERR"
        status_color = "" if t.success else "!"

        row = [
            pad(tid, widths[0]),
            pad(name, widths[1]),
            pad(start_str, widths[2]),
            pad(duration, widths[3], right=True),
            pad(tokens, widths[4], right=True),
            pad(status, widths[5]),
        ]
        line = "  ".join(row)
        if not t.success:
            typer.secho(line, fg=typer.colors.RED)
        else:
            typer.echo(line)


@app.command()
def view(trace_id: str = typer.Argument(..., help="Trace ID to inspect")) -> None:
    """Print a detailed view of a trace and its step tree."""
    if len(trace_id.strip()) < 8:
        typer.secho("Trace ID must be at least 8 characters.", fg=typer.colors.RED)
        raise typer.Exit(1)

    store = SQLiteStore()
    trace = store.get_trace(trace_id)

    if trace is None:
        typer.secho(f"Trace '{trace_id}' not found.", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.echo(f"Trace ID   : {trace.id}")
    typer.echo(f"Name       : {trace.name}")
    typer.echo(f"Started    : {trace.start.isoformat()}")
    typer.echo(f"Duration   : {trace.duration:.3f}s")
    typer.echo(f"Tokens     : {trace.total_tokens}")
    typer.echo(f"Success    : {'Yes' if trace.success else 'No'}")
    typer.echo("\nStep Tree:")
    typer.echo("─" * 60)

    def print_step(step: object, depth: int = 0) -> None:
        indent = "  " * depth
        duration = getattr(step, "duration_ms", 0.0)
        kind = getattr(step, "kind", "step")
        name = getattr(step, "name", "unknown")
        status = ""
        if getattr(step, "content", {}).get("status") == "failed":
            status = " [FAILED]"

        typer.echo(f"{indent}• {name} ({kind}){status} — {duration:.0f}ms")
        for child in getattr(step, "children", []):
            print_step(child, depth + 1)

    print_step(trace.root_step)


@app.command()
def clear() -> None:
    """Delete all stored traces after confirmation."""
    if not typer.confirm("Delete ALL traces from the local database?"):
        typer.echo("Cancelled.")
        raise typer.Exit(0)

    store = SQLiteStore()
    deleted = store.clear_all()
    typer.echo(f"✓ Cleared {deleted} trace(s).")


def main() -> None:
    """Entry point for the console script."""
    app()


if __name__ == "__main__":
    main()
