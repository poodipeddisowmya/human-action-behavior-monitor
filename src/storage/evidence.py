"""Proof images for logged activities.

While a person holds a label, a few annotated shots of them are kept in
memory (at most one per EVIDENCE_INTERVAL_S). When the activity is logged,
the first, middle and last of those shots are written as JPEGs under
data/evidence/<date>/ and linked from the action's database row, so the
dashboard can show what the person was doing.

Shots are crops around the person with context, their box, and a banner
naming the person, the activity and the time: evidence, not a full CCTV
archive. They are served only through the API (behind ACCESS_TOKEN when
set) and deleted after EVIDENCE_RETENTION_DAYS.
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from config import settings

log = logging.getLogger(__name__)

SHOTS_PER_EVENT = 3
_MAX_BUFFER = 24            # shots kept in memory per person (~1 MB at most)
_MAX_SIDE = 640             # longest side of a saved shot, px
_PAD = 0.35                 # context around the box, fraction of its size
_JPEG_QUALITY = 85

# Labels that are not an activity anyone needs proof of.
_SKIP = ("detecting",)


@dataclass
class _Track:
    label: str
    shots: list[tuple[float, bytes]] = field(default_factory=list)
    last_shot: float = 0.0


def render_shot(frame: np.ndarray, xyxy, person: str, label: str,
                when: datetime, conf: float | None = None) -> bytes | None:
    """An annotated, context-padded crop of one person as JPEG bytes."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    cx1, cy1 = max(0, int(x1 - bw * _PAD)), max(0, int(y1 - bh * _PAD))
    cx2, cy2 = min(w, int(x2 + bw * _PAD)), min(h, int(y2 + bh * _PAD))
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return None
    crop = frame[cy1:cy2, cx1:cx2].copy()
    s = min(1.0, _MAX_SIDE / max(crop.shape[:2]))
    if s < 1.0:
        crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)),
                          interpolation=cv2.INTER_AREA)
    # The person's box inside the crop.
    bx1, by1 = int((x1 - cx1) * s), int((y1 - cy1) * s)
    bx2, by2 = int((x2 - cx1) * s), int((y2 - cy1) * s)
    cv2.rectangle(crop, (bx1, by1), (bx2, by2), (80, 200, 120), 2)
    # Banner: who, what, when.
    text = f"{person}  {label}" + (f" {conf:.0%}" if conf else "") + f"  {when:%Y-%m-%d %H:%M:%S}"
    scale = max(0.4, min(0.6, crop.shape[1] / 900))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    bar = th + 10
    out = np.zeros((crop.shape[0] + bar, crop.shape[1], 3), np.uint8)
    out[bar:] = crop
    cv2.putText(out, text, (6, th + 4), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (235, 235, 235), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    return buf.tobytes() if ok else None


def pick(shots: list[tuple[float, bytes]], n: int = SHOTS_PER_EVENT) -> list[bytes]:
    """First, middle and last: the start, the course and the end of it."""
    if len(shots) <= n:
        return [b for _, b in shots]
    idx = sorted({round(i * (len(shots) - 1) / (n - 1)) for i in range(n)})
    return [shots[i][1] for i in idx]


class EvidenceCollector:
    """Per person, the shots of the label they are holding now.

    `observe` is called for every tracked person on every processed frame and
    is cheap when no shot is due. `take` hands over the chosen shots of the
    event that just closed and forgets them.
    """

    def __init__(self, interval_s: float | None = None, enabled: bool | None = None):
        self.interval_s = settings.evidence_interval_s if interval_s is None else interval_s
        self.enabled = settings.evidence_enabled if enabled is None else enabled
        self._tracks: dict[int, _Track] = {}
        self._lock = threading.Lock()

    def observe(self, person_id: int, label: str, confidence: float,
                frame: np.ndarray, xyxy, name: str | None = None) -> None:
        if not self.enabled or not label or label.startswith(_SKIP):
            return
        now = time.monotonic()
        with self._lock:
            t = self._tracks.get(person_id)
            if t is None or t.label != label:
                t = self._tracks[person_id] = _Track(label)
            if now - t.last_shot < self.interval_s:
                return
            t.last_shot = now
        who = f"{name} (P{person_id:03d})" if name else f"P{person_id:03d}"
        shot = render_shot(frame, xyxy, who.encode("ascii", "replace").decode(), label,
                           datetime.now(), confidence)
        if shot is None:
            return
        with self._lock:
            t = self._tracks.get(person_id)
            if t is None or t.label != label:
                return
            t.shots.append((now, shot))
            if len(t.shots) > _MAX_BUFFER:
                # Thin the middle, keep the first and the newest.
                del t.shots[1:3]

    def take(self, person_id: int, label: str) -> list[bytes]:
        with self._lock:
            t = self._tracks.get(person_id)
            if t is None or t.label != label:
                return []
            shots, t.shots = t.shots, []
        return pick(shots)

    def forget(self, person_id: int) -> None:
        with self._lock:
            self._tracks.pop(person_id, None)

    def reset(self) -> None:
        with self._lock:
            self._tracks.clear()


def evidence_root() -> Path:
    return settings.data_dir / "evidence"


def save(record_id: int, shots: list[bytes], when: datetime) -> list[str]:
    """Write one event's shots; returns their paths relative to evidence_root()."""
    day = evidence_root() / f"{when:%Y-%m-%d}"
    day.mkdir(parents=True, exist_ok=True)
    out = []
    for i, jpg in enumerate(shots):
        rel = f"{when:%Y-%m-%d}/{record_id}_{i}.jpg"
        (evidence_root() / rel).write_bytes(jpg)
        out.append(rel)
    return out


def evidence_urls(record) -> list[str]:
    """API URLs of an action row's proof images (served by /api/evidence)."""
    try:
        n = len(json.loads(record.evidence)) if record.evidence else 0
    except (TypeError, ValueError):
        n = 0
    return [f"/api/evidence/{record.id}/{i}" for i in range(n)]


def purge_old(retention_days: int | None = None, today: date | None = None) -> int:
    """Delete day folders older than the retention period; returns how many."""
    days = settings.evidence_retention_days if retention_days is None else retention_days
    if days <= 0:
        return 0
    cutoff = (today or date.today()) - timedelta(days=days)
    root, removed = evidence_root(), 0
    if not root.is_dir():
        return 0
    for d in root.iterdir():
        try:
            day = datetime.strptime(d.name, "%Y-%m-%d").date()
        except ValueError:
            continue
        if d.is_dir() and day < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    if removed:
        log.info(f"evidence: removed {removed} day folder(s) older than {days} days")
    return removed
