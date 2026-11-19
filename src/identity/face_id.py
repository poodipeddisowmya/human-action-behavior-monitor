"""Face recognition: which enrolled person a tracked person is.

Models (OpenCV Zoo, both licensed for commercial use, run by OpenCV itself):
  YuNet  face detection     MIT          models/face/face_detection_yunet_2023mar.onnx
  SFace  face recognition   Apache-2.0   models/face/face_recognition_sface_2021dec.onnx
(InsightFace's models are more common but licensed for non-commercial
research only, so they are not used.)

Only people enrolled in the gallery (known_people / face_samples tables, with
their consent) are ever named; everyone else is "Unknown". Measured on the
Jetson, 2026-11-19 (4 people, 14 s each): the same person scored cosine
0.45-0.96 across a clip when the face was clear, different people at most
0.27; matches dropped to 0.06-0.30 only on weak detections (score < 0.85,
faces < 40 px, head turned down). Hence the rules here:

  * only clear faces are used (detection score >= min_quality, width >= min_face_px);
  * a match is the best over the person's enrolled photos, and must beat the
    runner-up person by `margin`;
  * a tracked person is named when 2 of their last 3 clear looks agree, and
    keeps the name; a different name needs 3 agreeing looks in a row.

Runs on the CPU in its own thread (~25-80 ms to find faces, ~25 ms an
embedding on the Orin's CPU), a few looks a second in total: it never holds
up the video or the GPU models.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from config import settings

log = logging.getLogger(__name__)

DET_MODEL = "face/face_detection_yunet_2023mar.onnx"
REC_MODEL = "face/face_recognition_sface_2021dec.onnx"


def _cfg(key: str, default):
    return (settings.face_id or {}).get(key, default)


class FaceModels:
    """YuNet + SFace, shared by the live identifier and enrolment."""

    def __init__(self) -> None:
        d = settings.models_dir
        self._det = cv2.FaceDetectorYN.create(str(d / DET_MODEL), "", (320, 320), 0.6, 0.3, 50)
        self._rec = cv2.FaceRecognizerSF.create(str(d / REC_MODEL), "")
        self._lock = threading.Lock()

    @staticmethod
    def available() -> bool:
        d = settings.models_dir
        return (d / DET_MODEL).is_file() and (d / REC_MODEL).is_file()

    def faces(self, img: np.ndarray) -> np.ndarray:
        """Detected faces, rows of [x, y, w, h, 10 landmarks, score]; may be empty."""
        h, w = img.shape[:2]
        with self._lock:
            self._det.setInputSize((w, h))
            _, f = self._det.detect(img)
        return f if f is not None else np.zeros((0, 15), np.float32)

    def embed(self, img: np.ndarray, face: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(unit-length 128-d embedding, 112x112 aligned face image)."""
        with self._lock:
            aligned = self._rec.alignCrop(img, face)
            e = self._rec.feature(aligned).reshape(-1).astype(np.float32)
        return e / (np.linalg.norm(e) + 1e-9), aligned


def best_face(models: FaceModels, img: np.ndarray, min_quality: float, min_px: float):
    """The largest clear face in img, or (None, reason)."""
    faces = models.faces(img)
    if len(faces) == 0:
        return None, "no face found"
    f = max(faces, key=lambda r: r[2] * r[3])
    if f[-1] < min_quality:
        return None, f"face not clear enough ({f[-1]:.2f} < {min_quality})"
    if f[2] < min_px:
        return None, f"face too small ({int(f[2])} px < {int(min_px)} px)"
    return f, ""


@dataclass
class Gallery:
    """Enrolled people: one row per face sample."""
    person_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    names: dict[int, str] = field(default_factory=dict)
    embeddings: np.ndarray = field(default_factory=lambda: np.zeros((0, 128), np.float32))

    def match(self, e: np.ndarray, threshold: float, margin: float) -> tuple[int | None, float]:
        """(person id, similarity) of the best match, or (None, best similarity)."""
        if len(self.embeddings) == 0:
            return None, 0.0
        sims = self.embeddings @ e
        best: dict[int, float] = {}
        for pid, s in zip(self.person_ids.tolist(), sims.tolist()):
            best[pid] = max(best.get(pid, -1.0), s)
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        pid, s = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else -1.0
        if s >= threshold and s - runner >= margin:
            return pid, s
        return None, s


@dataclass
class _Track:
    votes: deque = field(default_factory=lambda: deque(maxlen=3))   # person id or None
    person: int | None = None
    challenger: int | None = None
    challenges: int = 0
    last_look: float = 0.0
    similarity: float = 0.0


class FaceIdentifier:
    """Names tracked people from the gallery, in a background thread."""

    def __init__(self) -> None:
        self.enabled = bool(_cfg("enabled", True)) and FaceModels.available()
        self.threshold = float(_cfg("threshold", 0.363))     # SFace's published cosine bar
        self.margin = float(_cfg("margin", 0.05))
        self.min_quality = float(_cfg("min_quality", 0.85))
        self.min_px = float(_cfg("min_face_px", 40))
        self.look_s = float(_cfg("look_interval_s", 0.5))     # until named
        self.recheck_s = float(_cfg("recheck_s", 4.0))        # once named
        self.gallery = Gallery()
        self._tracks: dict[int, _Track] = {}
        self._lock = threading.Lock()
        self._jobs: queue.Queue = queue.Queue(maxsize=4)
        self.models: FaceModels | None = None
        if not self.enabled:
            if not FaceModels.available():
                print("[face-id] models missing (models/face/); people stay unnamed", flush=True)
            return
        try:
            self.models = FaceModels()
        except Exception as e:
            self.enabled = False
            print(f"[face-id] unavailable ({e})", flush=True)
            return
        threading.Thread(target=self._worker, daemon=True, name="face-id").start()
        print("[face-id] ready (YuNet + SFace)", flush=True)

    # -- gallery -------------------------------------------------------------
    def set_gallery(self, gallery: Gallery) -> None:
        """Swap in a new gallery (after an enrolment or deletion): names
        already given are re-checked against it."""
        with self._lock:
            self.gallery = gallery
            for t in self._tracks.values():
                if t.person is not None and t.person not in gallery.names:
                    t.person = None
                t.votes.clear()
                t.last_look = 0.0

    # -- per frame -----------------------------------------------------------
    def observe(self, person_id: int, frame: np.ndarray, xyxy) -> None:
        if not self.enabled or len(self.gallery.embeddings) == 0:
            return
        now = time.monotonic()
        with self._lock:
            t = self._tracks.setdefault(person_id, _Track())
            due = self.recheck_s if t.person is not None else self.look_s
            if now - t.last_look < due:
                return
            t.last_look = now
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        bw, bh = x2 - x1, y2 - y1
        # The head is in the upper part of a person's box; a little margin
        # around it keeps the detector's context.
        crop = frame[max(0, int(y1 - 0.1 * bh)):min(h, int(y1 + 0.6 * bh)),
                     max(0, int(x1 - 0.15 * bw)):min(w, int(x2 + 0.15 * bw))]
        if crop.size == 0 or min(crop.shape[:2]) < self.min_px:
            return
        try:
            self._jobs.put_nowait((person_id, crop.copy()))
        except queue.Full:
            pass

    def _worker(self) -> None:
        while True:
            pid, crop = self._jobs.get()
            try:
                f, _ = best_face(self.models, crop, self.min_quality, self.min_px)
                if f is None:
                    continue                    # no clear face: not a vote either way
                e, _ = self.models.embed(crop, f)
                who, sim = self.gallery.match(e, self.threshold, self.margin)
                self._vote(pid, who, sim)
            except Exception as ex:
                print(f"[face-id] look failed: {ex}", flush=True)

    def _vote(self, pid: int, who: int | None, sim: float) -> None:
        with self._lock:
            t = self._tracks.get(pid)
            if t is None:
                return
            t.votes.append(who)
            if t.person is None:
                if who is not None and list(t.votes).count(who) >= 2:
                    t.person, t.similarity = who, sim
                    print(f"[face-id] P{pid} is {self.gallery.names.get(who)} ({sim:.2f})", flush=True)
            elif who is not None and who != t.person:
                # A different enrolled face, clearly: needs 3 in a row to switch.
                t.challenges = t.challenges + 1 if t.challenger == who else 1
                t.challenger = who
                if t.challenges >= 3:
                    t.person, t.challenger, t.challenges = who, None, 0
            elif who == t.person:
                t.challenger, t.challenges, t.similarity = None, 0, max(t.similarity, sim)

    def name(self, person_id: int) -> str | None:
        with self._lock:
            t = self._tracks.get(person_id)
            return self.gallery.names.get(t.person) if t and t.person is not None else None

    def forget(self, person_id: int) -> None:
        with self._lock:
            self._tracks.pop(person_id, None)

    def reset(self) -> None:
        with self._lock:
            self._tracks.clear()


def faces_dir() -> Path:
    return settings.data_dir / "faces"
