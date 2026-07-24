"""FastAPI application entry point.

Start the server:
    uv run fastapi dev main.py
    # or for production:
    uv run fastapi run main.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes.analyze import router as analyze_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Episodic Intelligence Engine",
    description=(
        "AI-powered API that decomposes story ideas into optimised "
        "multi-episode arcs for 90-second vertical video."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Simple liveness probe."""
    return {"status": "ok"}


# ── Serve React static files (production) ──────────────────────
if settings.static_dir:
    _static_dir = Path(settings.static_dir)

    if _static_dir.is_dir():
        # Serve hashed assets (JS/CSS/images) under /assets
        _assets_dir = _static_dir / "assets"
        if _assets_dir.is_dir():
            app.mount(
                "/assets",
                StaticFiles(directory=str(_assets_dir)),
                name="static-assets",
            )

        # Serve other root-level static files (favicon.ico, etc.)
        app.mount(
            "/static",
            StaticFiles(directory=str(_static_dir)),
            name="static-root",
        )

        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(request: Request, full_path: str):
            """SPA fallback: serve index.html for any non-API route."""
            # Try to serve an exact file match first (e.g. favicon.ico)
            file_path = _static_dir / full_path
            if full_path and file_path.is_file():
                return FileResponse(str(file_path))
            # Otherwise return the SPA entry point
            return FileResponse(str(_static_dir / "index.html"))

        logger.info("Serving frontend SPA from %s", _static_dir)
