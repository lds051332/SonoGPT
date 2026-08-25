"""Local FastAPI demo. Process-resident model, bind loopback by default."""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, ValidationError

from sonogpt.inference.engine import InferenceEngine
from sonogpt.web.catalog import WEB_DEMO_VERSION, catalog_payload
from sonogpt.web.demo import run_demo_generate, run_demo_template
from sonogpt.web.forms import coerce_exam

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"
FAVICON_ICO = STATIC_DIR / "favicon.ico"


class DemoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exam: dict[str, Any]
    fallback_template: bool = True


def create_app(
    engine: InferenceEngine,
    *,
    preload: bool = False,
) -> FastAPI:
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if preload:
            engine.info(load_model=True)
        yield

    app = FastAPI(
        title="SonoGPT local demo",
        version=WEB_DEMO_VERSION,
        summary="Learning demo only. Not for clinical diagnosis.",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.engine = engine
    app.state.generate_lock = lock

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        if not INDEX_HTML.is_file():
            raise HTTPException(status_code=500, detail="demo page is missing")
        return FileResponse(INDEX_HTML)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:
        if not FAVICON_ICO.is_file():
            raise HTTPException(status_code=404, detail="favicon is missing")
        return FileResponse(FAVICON_ICO, media_type="image/x-icon")

    @app.get("/api/health")
    def health() -> dict[str, object]:
        checkpoint_ready = engine.checkpoint_path.is_file()
        return {
            "ok": True,
            "clinical_use": False,
            "web_demo_version": WEB_DEMO_VERSION,
            "checkpoint_ready": checkpoint_ready,
            "checkpoint_name": engine.checkpoint_path.name
            if checkpoint_ready
            else None,
            "device": str(engine.device),
            "model_loaded": engine._model is not None,
        }

    @app.get("/api/meta")
    def meta() -> dict[str, object]:
        payload = catalog_payload()
        payload["health"] = health()
        payload["info"] = engine.info(load_model=False).to_dict()
        return payload

    def _exam_or_422(payload: dict[str, Any]):
        try:
            return coerce_exam(payload)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=error.errors()) from error

    @app.post("/api/template")
    def template(body: DemoRequest) -> dict[str, object]:
        exam = _exam_or_422(body.exam)
        return run_demo_template(exam)

    @app.post("/api/generate")
    def generate(request: Request, body: DemoRequest) -> dict[str, object]:
        exam = _exam_or_422(body.exam)
        demo_engine: InferenceEngine = request.app.state.engine
        demo_lock: threading.Lock = request.app.state.generate_lock
        try:
            with demo_lock:
                return run_demo_generate(
                    demo_engine,
                    exam,
                    fallback_template=body.fallback_template,
                )
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post("/api/extract")
    def extract(body: dict[str, Any]) -> dict[str, object]:
        report = body.get("report")
        if not isinstance(report, str) or not report.strip():
            raise HTTPException(status_code=422, detail="report text is required")
        result = engine.extract(report)
        result["web_demo_version"] = WEB_DEMO_VERSION
        result["clinical_use"] = False
        return result

    return app
