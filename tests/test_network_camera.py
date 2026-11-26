"""CCTV / IP cameras (src/input/network_camera.py), against a local MJPEG
server: the HTTP form of network camera, read by the same FFmpeg path as
RTSP. Checks connecting, newest-frame-only reading, "no signal" while the
camera is away, and reconnecting when it comes back."""
from __future__ import annotations

import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from src.input import network_camera as nc
from src.input.network_camera import StreamCapture, host_of, lookup, safe_url


def _jpeg(i: int) -> bytes:
    img = np.zeros((240, 320, 3), np.uint8)
    img[:, :] = (i * 7 % 255, 80, 160)
    return cv2.imencode(".jpg", img)[1].tobytes()


class _Camera:
    """A fake IP camera: multipart JPEG over HTTP at ~25 fps."""

    def __init__(self, port: int = 0):
        stop = self.stop_flag = threading.Event()

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                i = 0
                try:
                    while not stop.is_set():
                        data = _jpeg(i)
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(data)}\r\n\r\n".encode()
                                         + data + b"\r\n")
                        i += 1
                        time.sleep(0.04)
                except OSError:
                    pass

        ThreadingHTTPServer.allow_reuse_address = True
        self.server = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://viewer:secret@127.0.0.1:{self.port}/video.mjpg"

    def close(self):
        self.stop_flag.set()
        self.server.shutdown()
        self.server.server_close()


def _wait(cond, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return False


def test_password_never_shown():
    url = "rtsp://admin:Secr3t@192.168.1.64:554/Streaming/Channels/102"
    assert host_of(url) == "192.168.1.64"
    assert "Secr3t" not in safe_url(url) and "admin:***@192.168.1.64:554" in safe_url(url)


def test_lookup_by_name_or_address(monkeypatch):
    monkeypatch.setattr(nc.settings, "network_cameras", {"Reception": "rtsp://a/1"})
    monkeypatch.setattr(nc, "_extra", {})
    assert lookup("reception") == "stream:Reception"
    assert lookup("stream:Reception") == "stream:Reception"
    assert lookup("rtsp://a/1") == "stream:Reception"           # already known
    assert lookup("rtsp://b/2") == "stream:Camera"              # registered
    assert lookup("Pantry") is None


def test_reads_newest_frame_and_reconnects(monkeypatch):
    monkeypatch.setattr(nc, "_RETRY_S", (0.3,))
    cam = _Camera()
    cap = StreamCapture("Test", cam.url)
    try:
        assert _wait(lambda: cap.status()["state"] == "live"), cap.status()
        ok, img = cap.read()
        assert ok and img.shape == (240, 320, 3)
        # Newest frame only: after a pause, one read, not a backlog.
        time.sleep(0.5)
        assert cap.read()[0]
        t = time.time()
        assert cap.read()[0] and time.time() - t < 0.2   # next frame, ~40 ms
        assert cap.status()["host"] == "127.0.0.1"

        port = cam.port
        cam.close()                                      # the camera goes away
        assert _wait(lambda: cap.status()["state"] == "no signal"), cap.status()
        card = cap.no_signal_frame()
        assert card.shape == (240, 320, 3)

        cam = _Camera(port)                              # and comes back
        assert _wait(lambda: cap.status()["state"] == "live"), cap.status()
        assert cap.status()["reconnects"] >= 1
        assert cap.read()[0]
    finally:
        cap.release()
        cam.close()


def test_monitor_shows_no_signal_card_not_a_frozen_picture(monkeypatch):
    from src.input import SwappableFileSource
    from src.input.video_source import is_live, source_name
    monkeypatch.setattr(nc, "_RETRY_S", (0.3,))
    with socket.socket() as s:                   # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    monkeypatch.setattr(nc, "_extra", {"Gate": f"http://127.0.0.1:{dead}/video"})
    src = SwappableFileSource("stream:Gate")
    try:
        assert is_live(src.current) and source_name(src.current) == "Gate"
        frame = next(iter(src.frames()))
        assert frame.image.mean() < 60           # the dark "no signal" card
        assert src.camera_status["state"] in ("connecting", "no signal")
    finally:
        src.close()


def test_unplugged_usb_camera_plays_a_clip_and_comes_back(monkeypatch):
    """Started (e.g. by the service at boot) with its USB camera unplugged:
    the monitor shows a clip instead of exiting, and switches to the camera
    by itself once it is plugged in."""
    from src.input import video_source as vs

    class _Cam:
        def __init__(self):
            self.n = 0

        def read(self):
            self.n += 1
            return True, np.full((90, 160, 3), 200, np.uint8)

        def get(self, prop):
            return 30.0

        def release(self):
            pass

    plugged = {"yes": False}
    monkeypatch.setattr(vs, "open_camera", lambda i: _Cam() if plugged["yes"] else None)
    monkeypatch.setattr(vs, "_CAMERA_RETRY_S", 0.0)
    src = vs.SwappableFileSource("camera:0")
    try:
        assert not vs.is_live(src.current)                 # a demo clip meanwhile
        frames = src.frames()
        next(frames)
        assert not vs.is_live(src.current)                 # still unplugged
        plugged["yes"] = True
        next(frames)
        assert str(src.current) == "camera:0"              # picked up by itself
        assert next(frames).image.mean() == 200
    finally:
        src.close()
