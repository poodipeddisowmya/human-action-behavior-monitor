"""MJPEG stream of the annotated video, so the UI can render it in an <img>."""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import Response, StreamingResponse

from config import settings
from src.api.runtime import runtime

router = APIRouter()


async def _mjpeg() -> AsyncIterator[bytes]:
    """Newest frame, at most STREAM_FPS a second, never the same frame twice.

    Encoding runs in a worker thread (it is ~5-15 ms of CPU and would
    otherwise stall every other request on the event loop) and is shared by
    all viewers through runtime.latest_jpeg(). Skipping repeats matters over
    a tunnel: re-sending an unchanged frame is pure bandwidth.
    """
    period = 1 / max(1, settings.stream_fps)
    sent = -1
    while True:
        t0 = time.monotonic()
        if runtime.frame_seq != sent:
            got = await asyncio.to_thread(runtime.latest_jpeg)
            if got is not None and got[0] != sent:
                sent, jpg = got
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        await asyncio.sleep(max(0.005, period - (time.monotonic() - t0)))


@router.get("/video")
async def video_stream() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame",
        # Proxies and tunnels must pass frames straight through, not buffer.
        headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
    )


@router.get("/snapshot")
async def snapshot() -> Response:
    """The current annotated frame as one JPEG (thumbnails, monitoring)."""
    got = await asyncio.to_thread(runtime.latest_jpeg)
    if got is None:
        return Response(status_code=503)
    return Response(got[1], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/live")
async def live_state() -> dict:
    """Current per-person labels, exactly as drawn on the video right now.

    Distinct from /api/events, which only lists *closed* events. A person
    holding one steady label produces no closed event at all, so events alone
    cannot answer "is anyone labelled right now?" — the question that actually
    matters for monitoring and for verifying the pipeline is alive.
    """
    anns = runtime.latest_annotations or []
    caption = runtime.latest_caption or {}
    return {
        "tracked": len(anns),
        "caption": caption.get("text"),
        "scene": caption.get("scene"),
        "people": [
            {
                "person_id": f"P{a['person_id']:03d}",
                "label": a.get("label", "-"),
                "confidence": round(float(a.get("confidence", 0.0)), 3),
            }
            for a in anns
        ],
    }
