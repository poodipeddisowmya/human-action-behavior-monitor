"""FastAPI entrypoint. Serves REST + WebSocket, hosts the pipeline runtime."""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from src.input.video_source import browser_feed, is_browser
from src.storage import init_db
from src.utils import get_logger

from . import auth
from .routes import events as events_route
from .routes import evidence as evidence_route
from .routes import known_people as known_people_route
from .routes import persons as persons_route
from .routes import sources as sources_route
from .routes import stream as stream_route
from .routes import showcase as showcase_route
from .runtime import runtime
from .websocket import broadcaster, ws_router

log = get_logger(__name__)

__version__ = "1.0.0"

# Populated by run_pipeline.py (or any entrypoint) *before* uvicorn starts.
_pending_source_factory = None


def set_pipeline_source(factory) -> None:
    """Register a callable that returns a `VideoSource`. Must be set *before*
    `uvicorn.run(...)` — the lifespan below picks it up on startup.
    """
    global _pending_source_factory
    _pending_source_factory = factory


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    log.info("database ready")
    bcast_task = asyncio.create_task(broadcaster.run())
    purge_task = asyncio.create_task(_purge_evidence_daily())

    pipeline = None
    if _pending_source_factory is not None:
        # Import here to avoid pulling model weights at import time.
        from src.pipeline import Pipeline
        runtime.bind_loop(asyncio.get_running_loop())
        source = _pending_source_factory()
        runtime.source = source            # exposed so /api/sources can swap it
        pipeline = Pipeline(
            source,
            on_frame=runtime.on_frame,
            on_event=runtime.on_event,
        )
        runtime.pipeline = pipeline
        try:
            await known_people_route.refresh()      # enrolled faces, if any
        except Exception as e:
            log.warning(f"face gallery not loaded: {e}")
        pipeline.start()
        log.info(f"pipeline started on {settings.device} ({_gpu_name() or 'no GPU'})")
        if settings.access_token:
            log.info("access token required (ACCESS_TOKEN is set)")

    try:
        yield
    finally:
        if pipeline is not None:
            pipeline.stop()
        bcast_task.cancel()
        purge_task.cancel()


async def _purge_evidence_daily() -> None:
    """Delete proof images past EVIDENCE_RETENTION_DAYS: now, then hourly."""
    from src.storage.evidence import purge_old
    while True:
        try:
            await asyncio.to_thread(purge_old)
        except Exception as e:
            log.warning(f"evidence clean-up failed: {e}")
        await asyncio.sleep(3600)


app = FastAPI(
    title="Human Action & Behavior Monitor",
    version=__version__,
    lifespan=lifespan,
)

@app.middleware("http")
async def _fresh_ui(request, call_next):
    """Make browsers re-check the page itself on every load.

    Without this, a browser kept showing an older build of the dashboard
    after the UI was rebuilt (seen on the client machine: no Live camera
    button until a hard refresh), because index.html was served with no
    caching instructions and the browser chose to reuse its copy. The
    JS/CSS bundles have content-hashed names, so they can still be cached
    for good — a new build simply has new names.
    """
    response = await call_next(request)
    path = request.url.path
    if not path.startswith(("/api", "/ws", "/assets/")):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    elif path.startswith("/assets/"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response


# Registered after _fresh_ui, so it runs first (Starlette wraps in reverse):
# an unauthenticated request never reaches any route or static file.
app.middleware("http")(auth.middleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.ui_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(persons_route.router, prefix="/api/persons", tags=["persons"])
app.include_router(events_route.router, prefix="/api/events", tags=["events"])
app.include_router(evidence_route.router, prefix="/api/evidence", tags=["evidence"])
app.include_router(known_people_route.router, prefix="/api/known-people", tags=["known people"])
app.include_router(sources_route.router, prefix="/api/sources", tags=["sources"])
app.include_router(stream_route.router, prefix="/api/stream", tags=["stream"])
app.include_router(showcase_route.router, prefix="/api/showcase", tags=["showcase"])
app.include_router(ws_router)

_STARTED = time.time()
# A pipeline that has produced no frame for this long is wedged.
_STALL_S = 15.0


def _gpu_name() -> str | None:
    if not str(settings.device).startswith("cuda"):
        return None
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


@app.get("/api/health")
async def health() -> dict:
    """Liveness: the server process answers. No token needed (probes)."""
    return {"status": "ok", "version": __version__, "device": settings.device,
            "gpu": _gpu_name(), "fps_target": settings.target_fps}


@app.get("/api/ready")
async def ready() -> JSONResponse:
    """Readiness: models loaded AND frames flowing. 503 otherwise, so a
    load balancer, Docker or the Kaggle launcher can tell "up" from "working"."""
    p = runtime.pipeline
    if p is None:
        return JSONResponse({"ready": False, "reason": "pipeline not started"}, 503)
    age = time.time() - p.stats["last_frame_at"] if p.stats["last_frame_at"] else None
    live_wait = is_browser(getattr(runtime.source, "current", ""))   # waiting on a viewer is fine
    if not p.alive:
        return JSONResponse({"ready": False, "reason": "pipeline thread died"}, 503)
    if age is None or (age > _STALL_S and not live_wait):
        return JSONResponse({"ready": False, "reason": "no frames yet" if age is None
                             else f"no frame for {age:.0f}s"}, 503)
    return JSONResponse({"ready": True, "last_frame_age_s": round(age, 2)})


@app.get("/api/metrics")
async def metrics() -> dict:
    """Throughput, latency and GPU memory, for the dashboard header and ops."""
    p = runtime.pipeline
    stats = dict(p.stats) if p else {}
    out = {
        "version": __version__,
        "uptime_s": round(time.time() - _STARTED),
        "device": settings.device,
        "gpu": _gpu_name(),
        "source": str(getattr(runtime.source, "current", "")) or None,
        # Native frame rate of what is playing: playback_fps near this
        # value means real time.
        "source_fps": round(float(getattr(runtime.source, "fps", 0) or 0), 2),
        "viewers": broadcaster.client_count,
        "browser_camera_clients": browser_feed.clients,
        "deep_backend": settings.model.deep_backend,
        "detector_imgsz": settings.model.detector_imgsz,
        "pipeline_alive": bool(p and p.alive),
        **{k: v for k, v in stats.items() if k not in ("started_at", "last_frame_at")},
    }
    if out["gpu"]:
        try:
            import torch
            out["gpu_mem_mb"] = round(torch.cuda.memory_allocated() / 2**20)
            out["gpu_mem_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**20)
        except Exception:
            pass
    return out


# --- Serve the built UI from this same server -------------------------------
# Mounted LAST so it cannot shadow /api or /ws: those routers are already
# registered above and FastAPI matches in registration order.
#
# This exists so the app runs with Python alone. During development the UI is
# served by Vite on :5173 and proxies /api here, which means a second runtime
# (Node) and a second process. For a machine that just needs to *run* the
# thing, that is two extra ways to fail. The UI calls /api with relative URLs,
# so once it is served from this origin the proxy is unnecessary.
#
# Rebuild after changing anything under ui/src:   cd ui && npm run build
_UI_DIST = settings.root / "ui" / "dist"
if (_UI_DIST / "index.html").exists():
    from fastapi.staticfiles import StaticFiles

    # html=True serves index.html for unknown paths, which is what a
    # single-page app needs for client-side routing and deep links.
    app.mount("/", StaticFiles(directory=str(_UI_DIST), html=True), name="ui")
    log.info(f"serving UI from {_UI_DIST}")
else:
    log.warning(
        f"no built UI at {_UI_DIST} — API only. "
        f"Build it with: cd ui && npm install && npm run build"
    )
