"""WebSocket fan-out. Any pipeline callback can call `broadcaster.publish(...)`
and every connected client receives the message."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.input.video_source import browser_feed
from src.utils import get_logger

from .auth import websocket_allowed

log = get_logger(__name__)
ws_router = APIRouter()


class _Broadcaster:
    def __init__(self):
        self._clients: set[WebSocket] = set()
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)
        log.info(f"ws client connected, total={len(self._clients)}")

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._clients:
            self._clients.discard(ws)
            log.info(f"ws client disconnected, total={len(self._clients)}")

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def publish(self, msg: dict[str, Any]) -> None:
        if not self._clients and msg.get("type") == "frame_annotations":
            return          # nobody watching: don't fill the queue with frames
        try:
            self._queue.put_nowait(msg)
        except asyncio.QueueFull:
            log.warning("ws queue full, dropping message")

    async def run(self) -> None:
        while True:
            msg = await self._queue.get()
            try:
                payload = json.dumps(msg, default=str)
            except Exception as e:
                log.warning(f"ws message not serialisable, dropped: {e}")
                continue
            # Iterate a snapshot: a client connecting during one of the awaits
            # below would otherwise raise "set changed size during iteration"
            # and end this task, silently freezing every dashboard.
            dead: list[WebSocket] = []
            for ws in list(self._clients):
                try:
                    # One stalled viewer (a laptop gone to sleep behind a
                    # tunnel) must not hold up everyone else's updates.
                    await asyncio.wait_for(ws.send_text(payload), timeout=2.0)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.disconnect(ws)
                try:
                    await ws.close()
                except Exception:
                    pass


broadcaster = _Broadcaster()


@ws_router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    if not websocket_allowed(ws):
        await ws.close(code=1008)          # policy violation: no/invalid token
        return
    await broadcaster.connect(ws)
    try:
        while True:
            await ws.receive_text()   # keep-alive; ignore contents
    except WebSocketDisconnect:
        pass
    finally:
        broadcaster.disconnect(ws)


# Largest browser frame accepted. A 960 px JPEG is ~60-120 KB; anything far
# beyond that is not a webcam frame.
_MAX_CAMERA_FRAME = 4 * 1024 * 1024


def _decode(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img if img is not None and img.ndim == 3 else None


@ws_router.websocket("/ws/camera")
async def camera_endpoint(ws: WebSocket):
    """Receives the viewer's webcam as JPEG frames (binary messages).

    Each reply is a tiny text ack the page uses for flow control: it sends
    the next frame only after the previous one was acknowledged, so a slow
    uplink lowers the frame rate instead of building a backlog (which would
    show the person what they did seconds ago).
    """
    if not websocket_allowed(ws):
        await ws.close(code=1008)
        return
    await ws.accept()
    browser_feed.clients += 1
    log.info("browser camera connected")
    try:
        while True:
            data = await ws.receive_bytes()
            if len(data) <= _MAX_CAMERA_FRAME:
                img = await asyncio.to_thread(_decode, data)
                if img is not None:
                    browser_feed.push(img)
            await ws.send_text("ok")
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        browser_feed.clients -= 1
        log.info("browser camera disconnected")
