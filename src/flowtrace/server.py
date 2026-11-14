"""FastAPI application factory for the FlowTrace dashboard.

Mounts static assets, Jinja2 templates, and the API router.
Serves the single-page Swiss-minimal UI at the root path.
"""

from __future__ import annotations

import pathlib

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .api import router as api_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app with static files, templates,
        API routes, and the root dashboard handler.
    """
    app = FastAPI(
        title="FlowTrace",
        version="0.1.0",
        description="Local observability dashboard for LLM agent workflows",
        docs_url=None,
        redoc_url=None,
    )

    # Static files (CSS, JS, etc.)
    # Resolve relative to this module so the package works after installation.
    static_dir = pathlib.Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(static_dir)),
            name="static",
        )

    # Templates (expects templates/index.html next to src/ or inside package data)
    templates_dir = pathlib.Path(__file__).parent.parent / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))

    app.include_router(api_router)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        """Serve the primary Swiss-minimal single-page dashboard."""
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "version": "0.1.0",
            },
        )

    return app
