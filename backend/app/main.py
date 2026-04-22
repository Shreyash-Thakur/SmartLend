from __future__ import annotations

from datetime import datetime, timezone
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app.database import init_db
from app.routers.applications import router as applications_router
from app.services.ml_service import get_predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title="SmartLend Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(applications_router, prefix="/api")


@app.on_event("startup")
def startup_event() -> None:
    # Ensure DB tables exist and model is loaded once at startup.
    try:
        init_db()
        get_predictor()
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
def health() -> dict[str, str]:
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect root to the interactive API docs for convenience."""
    return RedirectResponse(url="/docs")
