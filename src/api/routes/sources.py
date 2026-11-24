"""Video source management — list clips in data/samples/ and swap live.

Enables mid-demo switching between different sample videos without
restarting the pipeline (and therefore without reloading model weights).
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from config import settings
from src.api.runtime import runtime
from src.input import SwappableFileSource
from src.input.video_source import (
    BROWSER_CAMERA,
    CAMERA_PREFIX,
    is_browser,
    is_camera,
    is_live,
    open_camera,
)
from src.input.network_camera import host_of, is_stream, stream_key, streams

router = APIRouter()

# Human-readable descriptions keyed by base filename (without extension).
# Each demo clip is chosen to exercise exactly one target category so the
# expected label is unambiguous during a client demo.
_DESCRIPTIONS: dict[str, str] = {
    # --- Target-category demo clips (expected label in brackets) ---
    "demo-eating":              "Person eating  →  expect: eating",
    "demo-drinking":            "Person drinking from a cup  →  expect: drinking",
    "demo-cooking":             "Cutting food in a kitchen  →  expect: cooking",
    "demo-cleaning-mopping":    "Mopping a floor  →  expect: cleaning",
    "demo-using-computer":      "Typing at a keyboard  →  expect: using computer",
    "demo-writing":             "Writing on a board  →  expect: writing",
    "demo-cycling":             "Riding a bicycle  →  expect: cycling",
    "demo-exercising-pushup":   "Push-ups  →  expect: push-up / exercising",
    "demo-exercising-squats":   "Body-weight squats  →  expect: exercising",
    "demo-exercising-pullup":   "Pull-ups  →  expect: exercising",

    # --- Basic posture / motion clips (pose classifier, no deep model) ---
    "demo-basic-sit":           "Person sitting down  →  expect: sitting",
    "demo-basic-stand":         "Person standing  →  expect: standing",
    "demo-basic-walk":          "Person walking  →  expect: walking",
    "demo-basic-run":           "Person running  →  expect: running",

    # --- HD stock footage (720p). Chosen from data/candidates/720p by the
    #     T4 benchmark (kaggle/gpu_benchmark.py); the other HD candidates
    #     are NOT this reliable — see README accuracy note. ---
    "hd-eating":                "Eating at a table (HD)  →  expect: eating",
    "hd-eating-2":              "Eating (HD)  →  expect: eating",
    "hd-drinking":              "Drinking from a glass (HD)  →  expect: drinking",
    "hd-drinking-2":            "Drinking (HD)  →  expect: drinking",
    "hd-smoking":               "Smoking (HD)  →  expect: smoking",
    "hd-smoking-2":             "Smoking (HD)  →  expect: smoking",
    "hd-cooking":               "Cooking at a stove (HD)  →  expect: cooking",
    "hd-swimming":              "Freestyle swimming, close-up (HD)  →  expect: swimming",
    "hd-swimming-2":            "Swimming in an outdoor pool (HD)  →  expect: swimming",
    "hd-swimming-3":            "Swimmer in a lane pool (HD)  →  expect: swimming",
    "hd-dancing":               "Solo contemporary dance (HD)  →  expect: dancing",
    "hd-dancing-2":             "Solo modern dance (HD)  →  expect: dancing",
    "hd-dancing-3":             "People dancing at a party (HD)  →  expect: dancing",
    "hd-dancing-4":             "Friends dancing at home (HD)  →  expect: dancing",

    # --- Multi-person detection / tracking scenes ---
    "street-surveillance":      "Street view — multiple pedestrians (tracking demo)",
    "office-people":            "Office corridor — full-body walking (tracking demo)",
    "store-aisle":              "Retail aisle — several people, mixed motion (tracking demo)",
}


class SwitchRequest(BaseModel):
    filename: str            # "store-aisle.mp4", "camera:0" (USB), "stream:Reception" (CCTV)


# Cameras found on this machine, probed once. Opening a camera is slow
# (around a second each on Windows) and a camera already in use cannot be
# opened a second time, so the list is built on first request and kept.
_cameras: list[dict] | None = None


def _camera_names() -> list[str]:
    """Friendly device names from Windows, in enumeration order, if available."""
    if os.name != "nt":
        return []
    import subprocess
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-PnpDevice -Class Camera,Image -PresentOnly -ErrorAction SilentlyContinue"
             " | Where-Object Status -eq OK | ForEach-Object FriendlyName"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def _list_cameras() -> list[dict]:
    global _cameras
    if _cameras is not None:
        return _cameras
    in_use = None
    if runtime.source is not None and is_camera(getattr(runtime.source, "current", "")):
        in_use = int(str(runtime.source.current)[len(CAMERA_PREFIX):])
    if not settings.probe_cameras:
        # Headless server: each probe costs ~1 s and usually finds nothing.
        # A camera it was started on (--source 0) is still listed, so the UI
        # shows it as playing and can switch back to it from a clip.
        _cameras = [] if in_use is None else [{
            "filename": f"{CAMERA_PREFIX}{in_use}", "name": "USB camera",
            "description": "Live camera"}]
        return _cameras
    found = []
    for i in range(4):
        if i == in_use:
            found.append(i)
            continue
        cap = open_camera(i)
        if cap is None:
            continue
        ok, _ = cap.read()
        cap.release()
        if ok:
            found.append(i)
    names = _camera_names()
    _cameras = [{
        "filename": f"{CAMERA_PREFIX}{i}",
        "name": names[k] if len(names) == len(found) else f"Camera {k + 1}",
        "description": "Live camera",
    } for k, i in enumerate(found)]
    return _cameras


def _network_cameras() -> list[dict]:
    """CCTV / IP cameras from .env, by name; their addresses stay here."""
    return [{"filename": stream_key(name), "name": name,
             "description": f"CCTV camera · {host_of(url)}", "kind": "cctv"}
            for name, url in streams().items()]


@router.get("")
async def list_sources() -> dict:
    samples = settings.data_dir / "samples"
    samples.mkdir(parents=True, exist_ok=True)

    clips: list[dict] = []
    for f in sorted(samples.iterdir()):
        if f.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
            continue
        stem = f.stem
        clips.append({
            "filename": f.name,
            "name": stem,
            "description": _DESCRIPTIONS.get(stem, ""),
            "size_kb": f.stat().st_size // 1024,
        })

    current = None
    if runtime.source is not None and hasattr(runtime.source, "current"):
        cur = str(runtime.source.current)
        current = cur if is_live(cur) else Path(cur).name

    return {"clips": clips, "cameras": _network_cameras() + _list_cameras(), "current": current,
            # Connection state of the CCTV camera playing, if one is.
            "camera_status": getattr(runtime.source, "camera_status", None),
            # The viewer's own webcam is always offered; the page decides
            # whether this browser can actually provide one (it needs HTTPS
            # or localhost, and a camera).
            "browser_camera": BROWSER_CAMERA}


@router.post("")
async def switch_source(req: SwitchRequest) -> dict:
    if runtime.source is None:
        raise HTTPException(500, "no source is bound; is the pipeline running?")
    if not isinstance(runtime.source, SwappableFileSource):
        raise HTTPException(400, "current source does not support live switching")

    if is_browser(req.filename):
        runtime.source.switch_to(req.filename)
        return {"switched_to": req.filename}

    if is_stream(req.filename):
        if req.filename not in {c["filename"] for c in _network_cameras()}:
            raise HTTPException(404, f"camera not found: {req.filename}")
        runtime.source.switch_to(req.filename)
        return {"switched_to": req.filename}

    if is_camera(req.filename):
        if req.filename not in {c["filename"] for c in _list_cameras()}:
            raise HTTPException(404, f"camera not found: {req.filename}")
        runtime.source.switch_to(req.filename)
        return {"switched_to": req.filename}

    path = settings.data_dir / "samples" / req.filename
    if not path.exists():
        raise HTTPException(404, f"file not found: {req.filename}")

    runtime.source.switch_to(path)
    return {"switched_to": path.name}


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_SIZE = 500 * 1024 * 1024        # 500 MB per upload
_ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@router.post("/upload")
async def upload_source(
    file: UploadFile = File(...),
    switch: bool = True,
) -> dict:
    """Accept a video upload, save it to data/samples/, and optionally switch."""
    if not file.filename:
        raise HTTPException(400, "missing filename")

    stem = Path(file.filename).stem
    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED_EXT:
        raise HTTPException(400, f"unsupported extension: {ext}")

    safe_stem = _SAFE.sub("_", stem)[:64] or "upload"
    samples = settings.data_dir / "samples"
    samples.mkdir(parents=True, exist_ok=True)

    # Stream to a temp file first, hashing as we go. We cannot know whether
    # this upload duplicates one already on disk until we've seen all of it,
    # and we must not clobber an existing clip while finding out.
    tmp = samples / f".{safe_stem}.{os.getpid()}.part"
    digest = hashlib.sha256()
    written = 0
    try:
        with tmp.open("wb") as out:
            while chunk := await file.read(1 << 20):     # 1 MB chunks
                written += len(chunk)
                if written > _MAX_SIZE:
                    raise HTTPException(
                        413, f"file exceeds {_MAX_SIZE // (1024*1024)} MB limit"
                    )
                digest.update(chunk)
                out.write(chunk)

        # Re-uploading the same clip used to append _1, _2, _3... forever,
        # because the old code only tested whether the *name* was taken. Match
        # on content instead: if these exact bytes are already here, reuse that
        # file. Only the same-stem family is checked (that is what collides on
        # re-upload); scripts/dedupe_samples.py sweeps the whole directory.
        incoming = digest.hexdigest()
        dest = None
        for existing in _stem_family(samples, safe_stem, ext):
            if existing.stat().st_size == written and _sha256(existing) == incoming:
                dest = existing
                break

        deduped = dest is not None
        if dest is None:
            dest = samples / f"{safe_stem}{ext}"
            n = 1
            while dest.exists():
                dest = samples / f"{safe_stem}_{n}{ext}"
                n += 1
            tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)

    switched = False
    if switch and isinstance(runtime.source, SwappableFileSource):
        runtime.source.switch_to(dest)
        switched = True

    return {
        "filename": dest.name,
        "size_kb": dest.stat().st_size // 1024,
        "switched": switched,
        "deduplicated": deduped,
    }


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _stem_family(samples: Path, safe_stem: str, ext: str) -> list[Path]:
    """Files a previous upload of this same name would have produced."""
    pattern = re.compile(rf"^{re.escape(safe_stem)}(_\d+)?{re.escape(ext)}$")
    return [f for f in samples.iterdir() if f.is_file() and pattern.match(f.name)]
