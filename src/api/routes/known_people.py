"""Known people: enrol, list and delete the people face recognition may name.

Behind ACCESS_TOKEN like the rest of /api. Every change is applied to the
running pipeline at once (no restart).
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.identity import gallery
from src.storage import ActionRecord, get_session
from src.storage.evidence import evidence_root

from ..runtime import runtime

router = APIRouter()
_MAX_PHOTO_BYTES = 10 * 2**20


class NewPerson(BaseModel):
    name: str


class FromEvidence(BaseModel):
    event_id: int
    index: int = 0


async def refresh() -> None:
    """Hand the running pipeline the current gallery."""
    faces = getattr(runtime.pipeline, "faces", None)
    if faces is not None:
        faces.set_gallery(await gallery.load_gallery())


def _bad(e: Exception):
    raise HTTPException(400, str(e))


@router.get("")
async def list_known() -> list[dict]:
    return [p | {"created_at": p["created_at"].isoformat() + "Z"} for p in await gallery.list_people()]


@router.post("")
async def create(body: NewPerson) -> dict:
    try:
        return await gallery.create_person(body.name)
    except gallery.EnrolError as e:
        _bad(e)


@router.post("/{person_id}/photos")
async def add_photos(person_id: int, files: list[UploadFile] = File(...)) -> dict:
    """Enrol one or more photos; returns what was added and why others were not."""
    added, refused = [], []
    for f in files:
        data = await f.read(_MAX_PHOTO_BYTES + 1)
        if len(data) > _MAX_PHOTO_BYTES:
            refused.append({"file": f.filename, "reason": "larger than 10 MB"})
            continue
        try:
            added.append(await gallery.add_sample(person_id, data))
        except gallery.EnrolError as e:
            refused.append({"file": f.filename, "reason": str(e)})
    if added:
        await refresh()
    return {"added": added, "refused": refused}


@router.post("/{person_id}/from-evidence")
async def add_from_evidence(person_id: int, body: FromEvidence) -> dict:
    """Enrol the face in one of a logged activity's proof images."""
    async with get_session() as s:
        rec = await s.get(ActionRecord, body.event_id)
    paths = json.loads(rec.evidence) if rec and rec.evidence else []
    if not 0 <= body.index < len(paths):
        raise HTTPException(404, "no such proof image")
    path = evidence_root() / paths[body.index]
    if not path.is_file():
        raise HTTPException(404, "proof image no longer kept")
    try:
        sample = await gallery.add_sample(person_id, Path(path).read_bytes())
    except gallery.EnrolError as e:
        _bad(e)
    await refresh()
    return sample


@router.delete("/{person_id}")
async def delete(person_id: int) -> dict:
    try:
        await gallery.delete_person(person_id)
    except gallery.EnrolError as e:
        raise HTTPException(404, str(e))
    await refresh()
    return {"deleted": person_id}


@router.delete("/{person_id}/photos/{sample_id}")
async def delete_photo(person_id: int, sample_id: int) -> dict:
    try:
        await gallery.delete_sample(person_id, sample_id)
    except gallery.EnrolError as e:
        raise HTTPException(404, str(e))
    await refresh()
    return {"deleted": sample_id}


@router.get("/{person_id}/photos/{sample_id}")
async def photo(person_id: int, sample_id: int) -> FileResponse:
    path = await gallery.sample_image(person_id, sample_id)
    if path is None:
        raise HTTPException(404, "no such photo")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
