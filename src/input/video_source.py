"""Unified video source abstraction.

Every downstream stage consumes `VideoSource`. Swapping a dataset clip for a
webcam or RTSP stream is a one-line change — no pipeline code is affected.
"""
from __future__ import annotations

import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .network_camera import STREAM_PREFIX, StreamCapture, is_stream, streams


@dataclass
class Frame:
    index: int
    timestamp: float          # seconds since stream start
    image: np.ndarray         # BGR, HxWx3


class VideoSource(ABC):
    @abstractmethod
    def frames(self) -> Iterator[Frame]: ...

    @property
    @abstractmethod
    def fps(self) -> float: ...

    def close(self) -> None:  # override if needed
        pass


class _OpenCVSource(VideoSource):
    def __init__(self, target: str | int, loop: bool = False):
        if isinstance(target, str) and not Path(target).exists():
            raise FileNotFoundError(
                f"Video file not found: {target}\n"
                f"Hint: run `python scripts\\get_sample_video.py` "
                f"or place any .mp4 at data\\samples\\demo.mp4"
            )
        self._target = target
        self._cap = cv2.VideoCapture(target)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {target}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._loop = loop

    @property
    def fps(self) -> float:
        return self._fps

    def frames(self) -> Iterator[Frame]:
        idx = 0
        while True:
            ok, img = self._cap.read()
            if not ok:
                if not self._loop:
                    break
                # Rewind to first frame and keep going — demos run indefinitely.
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, img = self._cap.read()
                if not ok:
                    break
            yield Frame(index=idx, timestamp=idx / self._fps, image=img)
            idx += 1

    def close(self) -> None:
        self._cap.release()


class FileSource(_OpenCVSource):
    def __init__(self, path: str | Path, loop: bool = True):
        # Demos loop by default so the UI keeps updating; pass loop=False for
        # one-shot evaluation runs.
        super().__init__(str(path), loop=loop)


class CameraSource(_OpenCVSource):
    """A webcam by index, read directly (no reconnects). The monitor itself
    uses SwappableFileSource, which also handles CCTV cameras."""
    def __init__(self, index_or_url: int | str = 0):
        super().__init__(index_or_url)


class DatasetSource(_OpenCVSource):
    """Reads a video clip out of a dataset directory (UCF101/Kinetics-style)."""
    def __init__(self, dataset_root: str | Path, clip: str):
        super().__init__(str(Path(dataset_root) / clip))


CAMERA_PREFIX = "camera:"
_CAMERA_RETRY_S = 10.0


def _first_clip() -> Path | None:
    """The clip shown while a USB camera is missing (as run_pipeline's default)."""
    samples = Path(__file__).resolve().parents[2] / "data" / "samples"
    preferred = samples / "demo-cooking.mp4"
    if preferred.exists():
        return preferred
    clips = sorted(samples.glob("*.mp4")) + sorted(samples.glob("*.avi"))
    return clips[0] if clips else None

# Live-camera capture settings. 960x540 rather than 1280x720: measured on a
# Logitech C270 indoors, 720p came in at ~10 fps (the camera lengthens its
# exposure in ordinary room light) against ~15 fps at 960x540, and every
# model downstream works at 416-640 px anyway. MJPG keeps USB bandwidth low.
_CAM_SIZE = (960, 540)
_CAM_FPS = 30


def is_camera(target) -> bool:
    return str(target).startswith(CAMERA_PREFIX)


# The viewer's own webcam, streamed from the browser over /ws/camera. This is
# what makes "Go live" work when the server has no camera of its own — a
# Kaggle notebook, a cloud GPU, a rack server: the camera is wherever the
# person watching the dashboard is.
BROWSER_CAMERA = "browser:camera"
_BROWSER_FPS = 15.0
# No frame from the browser for this long (tab closed, camera revoked,
# network dropped) -> go back to the clip that was playing, rather than
# freezing on the last frame forever.
_BROWSER_IDLE_S = 8.0


def is_browser(target) -> bool:
    return str(target) == BROWSER_CAMERA


def source_name(target) -> str | None:
    """What a logged activity was seen on: the CCTV camera's name, "USB
    camera", "Browser webcam", or the video file's name."""
    t = str(target or "")
    if is_stream(t):
        return t[len(STREAM_PREFIX):]
    if is_camera(t):
        return "USB camera"
    if is_browser(t):
        return "Browser webcam"
    return Path(t).name or None


def is_live(target) -> bool:
    return is_camera(target) or is_browser(target) or is_stream(target)


class BrowserFrameFeed:
    """Latest frame pushed by a browser; the pipeline thread waits on it."""

    def __init__(self):
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._at = 0.0
        self.clients = 0

    def push(self, image: np.ndarray) -> None:
        with self._cond:
            self._frame = image
            self._seq += 1
            self._at = time.monotonic()
            self._cond.notify_all()

    @property
    def seq(self) -> int:
        return self._seq

    @property
    def idle_s(self) -> float:
        return time.monotonic() - self._at if self._at else float("inf")

    def wait_next(self, after_seq: int, timeout: float) -> tuple[int, np.ndarray] | None:
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > after_seq, timeout=timeout):
                return None
            return self._seq, self._frame


browser_feed = BrowserFrameFeed()


class _BrowserCapture:
    """cv2.VideoCapture look-alike over browser_feed, so SwappableFileSource
    treats the browser webcam exactly like any other capture."""

    def __init__(self):
        self._seq = browser_feed.seq

    def isOpened(self) -> bool:
        return True

    def read(self):
        got = browser_feed.wait_next(self._seq, timeout=1.0)
        if got is None:
            return False, None
        self._seq, img = got
        return True, img

    def get(self, prop):
        return _BROWSER_FPS if prop == cv2.CAP_PROP_FPS else 0.0

    def set(self, *_):
        return False

    def release(self) -> None:
        pass


def open_camera(index: int):
    """Open a USB/built-in camera with settings tuned for live detection."""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAM_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAM_SIZE[1])
    cap.set(cv2.CAP_PROP_FPS, _CAM_FPS)
    # Two driver buffers: what is shown is at most one frame (33 ms) old, not
    # a backlog. Not one: with a single buffer the camera has nowhere to put
    # the next frame while we hold the current one, so it skips every other
    # frame. Measured on a Logitech C270 at 30 fps (Jetson Orin, 2026-11-17):
    # 1 buffer -> exactly 15.0 fps whatever the exposure.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    _hold_frame_rate(index)
    return cap


def _hold_frame_rate(index: int) -> None:
    """Stop a UVC webcam halving its frame rate in dim light (Linux).

    Logitech webcams ship with exposure_auto_priority on: in an ordinary
    indoor room they lengthen the exposure and deliver 15 fps instead of 30.
    Turning it off keeps 30 fps (the image is a little darker); measured on
    the Jetson with a C270, 2026-11-19: 15.0 -> 29.8 fps. OpenCV has no
    property for it, so set the V4L2 control directly. Best effort: a camera
    without the control is left as it is.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        import fcntl
        import os
        import struct
        fd = os.open(f"/dev/video{index}", os.O_RDWR)
        try:
            VIDIOC_S_CTRL, V4L2_CID_EXPOSURE_AUTO_PRIORITY = 0xC008561C, 0x009A0903
            fcntl.ioctl(fd, VIDIOC_S_CTRL, bytearray(struct.pack("Ii", V4L2_CID_EXPOSURE_AUTO_PRIORITY, 0)))
        finally:
            os.close(fd)
    except OSError:
        pass


class SwappableFileSource(VideoSource):
    """Video source that can switch to a different file — or to a live
    camera ("camera:0") — at runtime.

    Used by the API to change the demo clip mid-run without restarting the
    pipeline (and therefore without reloading model weights).
    """

    def __init__(self, initial_path: str | Path):
        self._current = self._key(initial_path)
        self._pending: Path | None = None
        # A USB camera that is unplugged: retried every _CAMERA_RETRY_S while a
        # clip plays (CCTV cameras reconnect by themselves, in StreamCapture).
        self._retry_camera: Path | None = None
        self._retry_at = 0.0
        # Last file played, so a live source that goes silent can return to it.
        self._last_file: Path | None = None if is_live(self._current) else self._current
        try:
            self._cap = self._open(self._current)
        except RuntimeError as e:
            if not is_camera(self._current):
                raise
            # Started at boot by the service with its camera unplugged: a
            # monitor that exits would only restart, exit again, and show
            # nothing; this one shows a clip and picks the camera up when
            # it is plugged in.
            self._wait_for_camera(self._current, str(e))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def _wait_for_camera(self, camera: Path, why: str) -> None:
        """Play a clip meanwhile; frames() retries the camera every few seconds."""
        fallback = self._last_file or _first_clip()
        if fallback is None:
            raise RuntimeError(why)
        print(f"[source] {why} (unplugged?): playing {fallback.name} meanwhile, "
              f"trying the camera again every {_CAMERA_RETRY_S:.0f} s", flush=True)
        self._retry_camera, self._retry_at = camera, time.monotonic() + _CAMERA_RETRY_S
        self._current = fallback
        self._cap = self._open(fallback)

    @staticmethod
    def _key(target) -> Path:
        # Path("camera:0") keeps the string intact, so `current` and the
        # API's source key work the same way for cameras and files.
        return Path(str(target))

    @staticmethod
    def _open(path: Path):
        if is_browser(path):
            return _BrowserCapture()
        if is_stream(path):
            name = str(path)[len(STREAM_PREFIX):]
            url = streams().get(name)
            if url is None:
                raise RuntimeError(f"Unknown network camera: {name} (add CAMERA_<NAME>= to .env)")
            return StreamCapture(name, url)
        if is_camera(path):
            cap = open_camera(int(str(path)[len(CAMERA_PREFIX):]))
            if cap is None:
                raise RuntimeError(f"Cannot open camera: {path}")
            return cap
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        return cap

    @property
    def current(self) -> Path:
        return self._current

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def camera_status(self) -> dict | None:
        """Connection state of the CCTV camera playing, if one is."""
        cap = self._cap
        return cap.status() if isinstance(cap, StreamCapture) else None

    def switch_to(self, new_path: str | Path) -> None:
        """Schedule a switch. Applied at the top of the next frame iteration."""
        self._pending = self._key(new_path)
        self._retry_camera = None                 # the operator's choice wins

    def frames(self) -> Iterator[Frame]:
        idx = 0
        t0 = time.monotonic()
        while True:
            if (self._retry_camera is not None and self._pending is None
                    and time.monotonic() >= self._retry_at):
                cap = open_camera(int(str(self._retry_camera)[len(CAMERA_PREFIX):]))
                if cap is None:
                    self._retry_at = time.monotonic() + _CAMERA_RETRY_S
                else:
                    print(f"[source] {self._retry_camera} is back", flush=True)
                    self._cap.release()
                    self._cap, self._current = cap, self._retry_camera
                    self._retry_camera = None
                    self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
                    idx, t0 = 0, time.monotonic()
            if self._pending is not None:
                previous = self._current
                self._cap.release()
                self._current = self._pending
                self._pending = None
                try:
                    self._cap = self._open(self._current)
                except Exception:
                    # A camera that is unplugged or busy must not take the
                    # whole monitor down: fall back to what was playing.
                    self._current = previous
                    self._cap = self._open(previous)
                self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
                if not is_live(self._current):
                    self._last_file = self._current
                idx = 0
                t0 = time.monotonic()

            ok, img = self._cap.read()
            if not ok and isinstance(self._cap, StreamCapture):
                # The CCTV camera is away and reconnecting by itself. Show
                # that, about once a second, rather than freeze on its last
                # picture as if all were well.
                if self._pending is not None:
                    continue
                ok, img = True, self._cap.no_signal_frame()
            if not ok:
                if is_browser(self._current):
                    # The browser stopped sending. Brief gaps are normal
                    # (a slow uplink); a long one means the viewer left.
                    if (self._pending is None and self._last_file is not None
                            and browser_feed.idle_s > _BROWSER_IDLE_S):
                        self._pending = self._last_file
                    continue
                if is_camera(self._current):
                    # A live feed has no "start" to rewind to; a dropped read
                    # is retried, and a camera that stays silent is reopened.
                    time.sleep(0.05)
                    if self._pending is None:
                        self._cap.release()
                        try:
                            self._cap = self._open(self._current)
                        except RuntimeError as e:          # unplugged
                            self._wait_for_camera(self._current, str(e))
                            self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
                            idx, t0 = 0, time.monotonic()
                    continue
                # loop the current file
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, img = self._cap.read()
                if not ok:
                    break
            # Files run on their own clock; a camera runs on the real one.
            ts = (time.monotonic() - t0) if is_live(self._current) else idx / self._fps
            yield Frame(index=idx, timestamp=ts, image=img)
            idx += 1

    def close(self) -> None:
        self._cap.release()
