"""CCTV and IP cameras: RTSP or HTTP video over the network.

Cameras are configured by name in .env, one line each:

    CAMERA_RECEPTION=rtsp://viewer:secret@192.168.1.64:554/Streaming/Channels/102
    CAMERA_PANTRY=rtsp://viewer:secret@192.168.1.65:554/stream2

and are then selected as "stream:Reception", "stream:Pantry": from the
dashboard's Source menu, or with `--source Reception`. The address, with its
password, stays on the server: the dashboard, the API, the database and the
log only ever see the name (and the camera's host, without credentials).
"""
from __future__ import annotations

import os
import re
import threading
import time
import urllib.parse

import cv2
import numpy as np

from config import settings
from src.utils import get_logger

log = get_logger(__name__)

STREAM_PREFIX = "stream:"

# Cameras given on the command line (--source rtsp://...), on top of the
# ones in .env.
_extra: dict[str, str] = {}

# Waiting this long for a camera to answer, or for its next frame, means it
# is gone: reconnect. Without a timeout a camera that drops off the network
# blocks the reader forever (FFmpeg's default).
_OPEN_TIMEOUT_MS = 10_000
_READ_TIMEOUT_MS = 10_000
# Pause between reconnection attempts, growing while the camera stays away.
_RETRY_S = (1, 2, 5, 10)
_NAME = re.compile(r"[^A-Za-z0-9 _-]+")


def is_stream(target) -> bool:
    return str(target).startswith(STREAM_PREFIX)


def streams() -> dict[str, str]:
    """Every network camera by name: .env first, then the command line."""
    return {**settings.network_cameras, **_extra}


def stream_key(name: str) -> str:
    return f"{STREAM_PREFIX}{name}"


def register_stream(url: str, name: str | None = None) -> str:
    """Make a camera given by address selectable; returns its source key."""
    for known, u in streams().items():
        if u == url:
            return stream_key(known)
    name = _NAME.sub("", name or "").strip() or "Camera"
    base, n = name, 2
    while name in streams():
        name, n = f"{base} {n}", n + 1
    _extra[name] = url
    return stream_key(name)


def lookup(target: str) -> str | None:
    """'Reception', 'stream:Reception' or a camera's address -> its key."""
    t = str(target)
    if "://" in t:
        return register_stream(t)
    name = t[len(STREAM_PREFIX):] if is_stream(t) else t
    for known in streams():
        if known.lower() == name.lower():
            return stream_key(known)
    return None


def host_of(url: str) -> str:
    """The camera's address without user name or password."""
    p = urllib.parse.urlsplit(url)
    return p.hostname or "?"


def safe_url(url: str) -> str:
    """The address with the password hidden, for logs."""
    p = urllib.parse.urlsplit(url)
    if not p.password:
        return url
    netloc = f"{p.username}:***@{p.hostname}" + (f":{p.port}" if p.port else "")
    return urllib.parse.urlunsplit((p.scheme, netloc, p.path, p.query, p.fragment))


class StreamCapture:
    """A network camera read in its own thread, keeping only the newest frame.

    Two things a plain cv2.VideoCapture gets wrong on CCTV:

    - Lag that never recovers. FFmpeg queues the camera's frames; whenever
      the models take longer than a frame interval, the queue grows and the
      dashboard falls seconds, then minutes, behind real time. Reading
      continuously here and handing over only the newest frame keeps the
      delay at one frame, whatever the models cost.
    - A camera that drops (network blip, reboot, power cut) never comes
      back. Here it is reconnected, with a growing pause, for as long as it
      is selected; meanwhile `read()` returns False and the source shows a
      "no signal" card instead of a frozen picture.

    Looks like cv2.VideoCapture to SwappableFileSource.
    """

    def __init__(self, name: str, url: str):
        # TCP, not RTSP's default UDP: on a busy LAN or Wi-Fi, lost UDP
        # packets smear the picture for a second at a time.
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        self.name = name
        self._url = url
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._taken = 0
        self._fps = 25.0
        self._size = (960, 540)
        self._stop = threading.Event()
        self.connected = False
        self.error: str | None = None
        self.since = time.time()          # of the current state
        self.reconnects = 0
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"camera-{name}")
        self._thread.start()

    # ---- reader thread ----
    def _run(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, _OPEN_TIMEOUT_MS,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, _READ_TIMEOUT_MS])
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                if 1 <= fps <= 60:        # some cameras report 0 or 90000
                    self._fps = fps
                log.info(f"camera {self.name}: connected ({safe_url(self._url)})")
                attempt = 0
                while not self._stop.is_set():
                    ok, img = cap.read()
                    if not ok:
                        break
                    with self._cond:
                        if not self.connected:
                            self.connected, self.error, self.since = True, None, time.time()
                        self._frame = img
                        self._size = (img.shape[1], img.shape[0])
                        self._seq += 1
                        self._cond.notify_all()
                self._down("the stream stopped")
            else:
                self._down("cannot connect")
            cap.release()
            if self._stop.is_set():
                break
            wait = _RETRY_S[min(attempt, len(_RETRY_S) - 1)]
            log.warning(f"camera {self.name}: {self.error}; retrying in {wait} s")
            attempt += 1
            self.reconnects += 1
            self._stop.wait(wait)

    def _down(self, why: str) -> None:
        with self._cond:
            if self.connected or self.error != why:
                self.since = time.time()
            self.connected, self.error = False, why
            self._cond.notify_all()

    # ---- cv2.VideoCapture look-alike ----
    def isOpened(self) -> bool:
        return True

    def read(self, timeout: float = 1.0):
        """The newest frame not yet returned; (False, None) after `timeout`."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != self._taken or self._stop.is_set(),
                                timeout=timeout)
            if self._seq == self._taken or self._frame is None:
                return False, None
            self._taken = self._seq
            return True, self._frame

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._size[0])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._size[1])
        return 0.0

    def set(self, *_):
        return False

    def release(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        # A read blocked on a silent camera ends at its own timeout; the
        # thread is a daemon, so nothing waits for it.
        self._thread.join(timeout=0.5)

    # ---- status ----
    def status(self) -> dict:
        state = "live" if self.connected else ("connecting" if self.error is None else "no signal")
        return {"camera": self.name, "host": host_of(self._url), "state": state,
                "error": self.error, "since_s": round(time.time() - self.since, 1),
                "reconnects": self.reconnects}

    def no_signal_frame(self) -> np.ndarray:
        """What the dashboard shows while the camera is away."""
        w, h = self._size
        img = np.full((h, w, 3), 24, np.uint8)
        s = max(0.6, w / 1280)
        state = "Connecting..." if self.error is None else f"No signal: {self.error}, reconnecting"
        for text, y, scale, colour in (
                (self.name, 0.45, 1.4 * s, (235, 235, 235)),
                (state, 0.55, 0.8 * s, (90, 170, 255)),
                (f"{host_of(self._url)}  ·  {time.strftime('%H:%M:%S')}", 0.63, 0.6 * s, (150, 150, 150))):
            text = text.replace("·", "-")
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            cv2.putText(img, text, ((w - tw) // 2, int(h * y)), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, colour, 2, cv2.LINE_AA)
        return img
