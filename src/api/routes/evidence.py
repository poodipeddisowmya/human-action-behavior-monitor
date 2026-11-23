"""Proof images of logged activities (src/storage/evidence.py).

Looked up by action id and index, never by a path from the request, so no
file outside data/evidence/ can be reached. Behind ACCESS_TOKEN like the
rest of /api when it is set.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.storage import ActionRecord, get_session
from src.storage.evidence import evidence_root

router = APIRouter()


@router.get("/{record_id}/{index}")
async def evidence_image(record_id: int, index: int) -> FileResponse:
    async with get_session() as s:
        record = await s.get(ActionRecord, record_id)
    try:
        paths = json.loads(record.evidence) if record and record.evidence else []
    except ValueError:
        paths = []
    if not 0 <= index < len(paths):
        raise HTTPException(404, "no such image")
    root = evidence_root().resolve()
    path = (root / paths[index]).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(404, "image no longer kept")    # removed by retention
    return FileResponse(path, media_type="image/jpeg",
                        headers={"Cache-Control": "private, max-age=86400"})
