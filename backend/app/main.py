from __future__ import annotations

from datetime import datetime, timezone
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from backend.app.database import init_db
from backend.app.routers.agent import router as agent_router
from backend.app.routers.applications import router as applications_router
from backend.app.routers.customers import router as customers_router
from backend.app.routers.public import router as public_router
from backend.app.routers.relearning import router as relearning_router
from backend.app.routers.tabpfn import router as tabpfn_router
from backend.app.routers.voice import router as voice_router
from backend.app.services.ml_service import get_predictor
from backend.app.services.public_api_service import seed_recent_applications

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title="SmartLend Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(public_router)
app.include_router(agent_router, prefix="/api")
app.include_router(applications_router, prefix="/api")
app.include_router(customers_router, prefix="/api")
app.include_router(relearning_router, prefix="/api")
app.include_router(tabpfn_router, prefix="/api")
app.include_router(voice_router, prefix="/api")


@app.on_event("startup")
def startup_event() -> None:
    # Ensure DB tables exist and model is loaded once at startup.
    try:
        init_db()
        get_predictor()
        seed_recent_applications()
        logging.getLogger(__name__).info("Startup complete | DB initialized and model loaded")
    except Exception:
        logging.getLogger(__name__).exception("Startup initialization failed")
        raise


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        payload = detail
    else:
        payload = {"error": f"HTTP_{exc.status_code}", "details": str(detail)}
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logging.getLogger(__name__).exception("Unhandled server error")
    return JSONResponse(status_code=500, content={"error": "Internal server error", "details": str(exc)})


@app.get("/api/health")
def api_health() -> dict[str, object]:
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/health")
def health() -> dict[str, object]:
    # Same live payload as the public router's /health (which shadows this
    # route in practice): the serving artifact's real identity, never the old
    # hardcoded synthetic-era numbers.
    from backend.app.services.public_api_service import get_health_payload

    return get_health_payload()


# ---------------------------------------------------------------------------
# Static frontend (production/Docker). Starlette matches routes in
# registration order, so this mount comes LAST: every /api route above wins,
# and everything else falls through to the built React app. Client-side
# routes (/review, /dashboard, ...) are not files on disk, so 404s fall back
# to index.html and React Router resolves them in the browser.
# ---------------------------------------------------------------------------
import os
from pathlib import Path

from fastapi.staticfiles import StaticFiles

_STATIC_DIR = Path(
    os.environ.get("SMARTLEND_STATIC_DIR", "").strip()
    or Path(__file__).resolve().parents[2] / "frontend" / "dist"
)


class _SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):  # type: ignore[override]
        # Starlette signals a missing file either as a 404 response (old) or a
        # raised HTTPException (new); both mean "client-side route" here.
        from starlette.exceptions import HTTPException as _StarletteHTTPException

        try:
            response = await super().get_response(path, scope)
        except _StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise
            return await super().get_response("index.html", scope)
        if response.status_code == 404:
            response = await super().get_response("index.html", scope)
        return response


if _STATIC_DIR.is_dir():
    app.mount("/", _SPAStaticFiles(directory=_STATIC_DIR, html=True), name="frontend")
else:
    # Dev machines run the Vite server instead; keep the old convenience redirect.
    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """Redirect root to the interactive API docs for convenience."""
        return RedirectResponse(url="/docs")
