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


def _extra_keys() -> set[str]:
    f = _root() / "showcase.json"
    if not f.exists():
        return set()
    data = json.loads(f.read_text(encoding="utf-8"))
    return {x["key"] for x in data.get("extra_sides", [])}


@router.get("/video/{side}/{name}")
async def video(side: str, name: str) -> FileResponse:
    # left = before/, right = the folder itself, an extra side = its key's folder.
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "bad path")
    if side == "left":
        folder = "before"
    elif side == "right":
        folder = ""
    elif side in _extra_keys():
        folder = side
    else:
        raise HTTPException(400, "bad path")
    path = _root() / folder / name
    if not path.is_file() or path.suffix.lower() != ".mp4":
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="video/mp4")
