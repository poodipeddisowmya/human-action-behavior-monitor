"""Known people: enrol, list and delete the people face recognition may name.

Behind ACCESS_TOKEN like the rest of /api. Every change is applied to the
running pipeline at once (no restart).
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from sqlalchemy import select, update

from src.identity import gallery
from src.storage import ActionRecord, Person, get_session
from src.storage.evidence import evidence_root

from ..runtime import runtime

router = APIRouter()
_MAX_PHOTO_BYTES = 10 * 2**20


class NewPerson(BaseModel):
    name: str


class FromEvidence(BaseModel):
    event_id: int
    index: int = 0


class FromVideo(BaseModel):
    track: int                 # tracking number: 6 for "P006"


def _faces():
    faces = getattr(runtime.pipeline, "faces", None)
    if faces is None or not faces.enabled:
        raise HTTPException(503, "face recognition is off")
    return faces


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


@router.get("/on-screen")
async def on_screen() -> list[dict]:
    """People in view now with a clear face and no name, for naming.

    Their faces are held in memory only while they are tracked (see
    src/identity/face_id.py); nothing about them is stored until someone
    enrols them.
    """
    faces = getattr(runtime.pipeline, "faces", None)
    if faces is None or not faces.enabled:
        return []
    return [u | {"person_id": f"P{u['track']:03d}"} for u in faces.unnamed()]


@router.get("/on-screen/{track}/face")
async def on_screen_face(track: int) -> Response:
    shot = _faces().face_shot(track)
    if shot is None:
        raise HTTPException(404, "that person is no longer in view")
    return Response(shot, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.post("/{person_id}/from-video")
async def add_from_video(person_id: int, body: FromVideo) -> dict:
    """Enrol a person in view from their best faces, name them at once, and
    put the name on what they have done since they came into view."""
    faces = _faces()
    held, since = faces.faces_of(body.track)
    try:
        added = await gallery.add_faces(person_id, held)
    except gallery.EnrolError as e:
        _bad(e)
    await refresh()
    faces.assign(body.track, person_id)
    named = 0
    name = faces.gallery.names.get(person_id)
    if since is not None and name:
        async with get_session() as s:
            ext = f"P{body.track:03d}"
            pid = (await s.execute(select(Person.id).where(Person.external_id == ext))).scalar_one_or_none()
            if pid is not None:
                r = await s.execute(update(ActionRecord)
                                    .where(ActionRecord.person_id == pid,
                                           ActionRecord.start_time >= since - timedelta(seconds=2),
                                           ActionRecord.identity.is_(None))
                                    .values(identity=name))
                named = r.rowcount or 0
                await s.commit()
    return {"added": added, "events_named": named}


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
