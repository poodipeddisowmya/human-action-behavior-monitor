"""The recorded CPU-vs-GPU comparison, for the dashboard's "CPU vs GPU" tab.

Reads data/showcase/showcase.json (installed by scripts/install_showcase.py)
and serves its videos from data/showcase/. Recorded, not live: the two sides
ran on different machines (a laptop CPU and a GPU), on the same clips, with
the same software.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import settings
from src.utils.showcase import summarize

router = APIRouter()


def _root():
    return settings.data_dir / "showcase"


@router.get("")
async def showcase() -> dict:
    f = _root() / "showcase.json"
    if not f.exists():
        return {"available": False}
    return {"available": True, **summarize(json.loads(f.read_text(encoding="utf-8")))}


@router.get("/video/{side}/{name}")
async def video(side: str, name: str) -> FileResponse:
    if side not in ("left", "right") or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "bad path")
    path = _root() / ("before" if side == "left" else "") / name
    if not path.is_file() or path.suffix.lower() != ".mp4":
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="video/mp4")
