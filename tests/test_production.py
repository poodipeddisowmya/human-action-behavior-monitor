"""Production behaviour added for remote GPU deployment (Kaggle / cloud).

Each test pins down a failure that matters once the dashboard is reachable
from outside the machine: an unprotected public link, a dashboard that freezes
when a viewer joins, a browser camera that leaves the video stuck forever.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from config import settings
from src.api.main import app
from src.api.runtime import Runtime
from src.api.websocket import _Broadcaster
from src.input import video_source
from src.input.video_source import BROWSER_CAMERA, SwappableFileSource, browser_feed


# ------------------------------------------------------------------ auth
@pytest.fixture
def locked(monkeypatch):
    monkeypatch.setattr(settings, "access_token", "s3cret")
    return TestClient(app, follow_redirects=False)


def test_open_when_no_token_configured(monkeypatch):
    monkeypatch.setattr(settings, "access_token", "")
    assert TestClient(app).get("/api/stream/live").status_code == 200


def test_health_needs_no_token(locked):
    # Docker / tunnel / load-balancer probes have no credentials.
    assert locked.get("/api/health").status_code == 200


def test_api_rejects_missing_or_wrong_token(locked):
    assert locked.get("/api/stream/live").status_code == 401
    assert locked.get("/api/stream/live", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_bearer_header_accepted(locked):
    r = locked.get("/api/stream/live", headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200


def test_page_without_token_shows_login_not_dashboard(locked):
    r = locked.get("/")
    assert r.status_code == 401
    assert "Access token" in r.text


def test_token_link_sets_cookie_and_strips_token_from_url(locked):
    r = locked.get("/?token=s3cret")
    assert r.status_code == 303
    assert r.headers["location"] == "/"
    assert "ham_token=s3cret" in r.headers["set-cookie"]
    # The cookie alone is now enough — this is what the <img> stream and the
    # WebSocket rely on, since neither can send headers.
    locked.cookies.set("ham_token", "s3cret")
    assert locked.get("/api/stream/live").status_code == 200


def test_websocket_rejected_without_token(locked):
    with pytest.raises(WebSocketDisconnect):
        with locked.websocket_connect("/ws") as ws:
            ws.receive_text()


# ------------------------------------------------------------------ stream
def test_jpeg_encoded_once_per_frame_and_width_capped(monkeypatch):
    monkeypatch.setattr(settings, "stream_max_width", 640)
    rt = Runtime()
    assert rt.latest_jpeg() is None
    rt.on_frame(np.zeros((1080, 1920, 3), np.uint8), [])
    a = rt.latest_jpeg()
    assert a is rt.latest_jpeg()                 # shared, not re-encoded
    import cv2
    img = cv2.imdecode(np.frombuffer(a[1], np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 640 and img.shape[0] == 360
    rt.on_frame(np.zeros((100, 100, 3), np.uint8), [])
    assert rt.latest_jpeg()[0] == a[0] + 1


# ------------------------------------------------------------------ broadcaster
def test_client_joining_mid_broadcast_does_not_kill_broadcaster():
    """A viewer connecting while a message is being sent used to raise
    'set changed size during iteration' and end the broadcast task."""

    async def scenario():
        b = _Broadcaster()
        got = []

        class Slow:
            async def send_text(self, payload):
                b._clients.add(Fast())           # someone joins mid-send
                await asyncio.sleep(0)
                got.append(payload)

        class Fast:
            async def send_text(self, payload):
                got.append(payload)

        b._clients.add(Slow())
        task = asyncio.create_task(b.run())
        b.publish({"type": "event", "n": 1})
        b.publish({"type": "event", "n": 2})
        await asyncio.sleep(0.05)
        assert not task.done(), task.exception() if task.done() else None
        task.cancel()
        return got

    assert len(asyncio.run(scenario())) >= 2


# ------------------------------------------------------------------ browser camera
def _clip(tmp_path):
    import cv2
    p = tmp_path / "clip.avi"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(5):
        w.write(np.full((48, 64, 3), i * 40, np.uint8))
    w.release()
    return p


def test_browser_camera_frames_flow_through_swappable_source(tmp_path):
    src = SwappableFileSource(_clip(tmp_path))
    frames = src.frames()
    next(frames)
    src.switch_to(BROWSER_CAMERA)
    pushed = np.full((48, 64, 3), 7, np.uint8)

    def push_soon():
        time.sleep(0.2)
        browser_feed.push(pushed)

    import threading
    threading.Thread(target=push_soon, daemon=True).start()
    f = next(frames)
    assert str(src.current) == BROWSER_CAMERA
    assert (f.image == 7).all()


def test_silent_browser_camera_falls_back_to_last_clip(tmp_path, monkeypatch):
    monkeypatch.setattr(video_source, "_BROWSER_IDLE_S", 0.0)
    clip = _clip(tmp_path)
    src = SwappableFileSource(clip)
    frames = src.frames()
    next(frames)
    src.switch_to(BROWSER_CAMERA)
    f = next(frames)                             # no browser frames ever arrive
    assert str(src.current) == str(clip)
    assert f.image.shape == (48, 64, 3)
