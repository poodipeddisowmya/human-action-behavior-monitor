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
    """Webcam or RTSP. Placeholder for the live-camera phase."""
    def __init__(self, index_or_url: int | str = 0):
        super().__init__(index_or_url)


class DatasetSource(_OpenCVSource):
    """Reads a video clip out of a dataset directory (UCF101/Kinetics-style)."""
    def __init__(self, dataset_root: str | Path, clip: str):
        super().__init__(str(Path(dataset_root) / clip))


CAMERA_PREFIX = "camera:"

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


def is_live(target) -> bool:
    return is_camera(target) or is_browser(target)


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
    # Hold at most one frame in the driver queue, so what is shown is what
    # the camera sees now, not a backlog from a second ago.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class SwappableFileSource(VideoSource):
    """Video source that can switch to a different file — or to a live
    camera ("camera:0") — at runtime.

    Used by the API to change the demo clip mid-run without restarting the
    pipeline (and therefore without reloading model weights).
    """

    def __init__(self, initial_path: str | Path):
        self._current = self._key(initial_path)
        self._pending: Path | None = None
        self._cap = self._open(self._current)
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Last file played, so a live source that goes silent can return to it.
        self._last_file: Path | None = None if is_live(self._current) else self._current

    @staticmethod
    def _key(target) -> Path:
        # Path("camera:0") keeps the string intact, so `current` and the
        # API's source key work the same way for cameras and files.
        return Path(str(target))

    @staticmethod
    def _open(path: Path):
        if is_browser(path):
            return _BrowserCapture()
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

    def switch_to(self, new_path: str | Path) -> None:
        """Schedule a switch. Applied at the top of the next frame iteration."""
        self._pending = self._key(new_path)

    def frames(self) -> Iterator[Frame]:
        idx = 0
        t0 = time.monotonic()
        while True:
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
                        self._cap = self._open(self._current)
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
