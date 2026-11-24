"""Recorded demo videos (data/office-demo/): what the dashboard showed, saved
by scripts/record_live.py, with a short description of each in
recordings.json next to them."""
from __future__ import annotations

import json
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import settings

router = APIRouter()

_NAME = re.compile(r"^[A-Za-z0-9._-]+\.mp4$")


def _root():
    return settings.data_dir / "office-demo"


@router.get("")
async def recordings() -> dict:
    root = _root()
    meta_file = root / "recordings.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
    items = []
    for entry in meta.get("videos", []):
        f = root / entry["file"]
        # A Git LFS pointer (clone without LFS) is not a video.
        if f.is_file() and f.stat().st_size > 1024:
            items.append({**entry, "url": f"/api/recordings/{entry['file']}"})
    return {"recorded": meta.get("recorded"), "device": meta.get("device"),
            "how": meta.get("how"), "videos": items}


@router.get("/{name}")
async def video(name: str) -> FileResponse:
    path = _root() / name
    if not _NAME.match(name) or not path.is_file():
        raise HTTPException(404, "no such recording")
    return FileResponse(path, media_type="video/mp4")
