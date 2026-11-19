"""Enrolment: add, list and delete the people face recognition may name.

A sample is a photo with exactly one clear face (several similar-sized faces
are refused: it would be a guess which one is meant). Its 128-d SFace
embedding goes to face_samples, the 112x112 aligned face to data/faces/
(git-ignored; biometric data) for the dashboard's thumbnails. Deleting a
person deletes every sample and image. Enrol people only with their consent.
"""
from __future__ import annotations

import shutil

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.storage import FaceSample, KnownPerson, get_session

from .face_id import FaceModels, Gallery, _cfg, best_face, faces_dir

ENROL_MIN_PX = 60          # enrolment photos must show the face at least this wide
_models: FaceModels | None = None


class EnrolError(ValueError):
    """A photo that cannot be enrolled; the message says why, for the user."""


def models() -> FaceModels:
    global _models
    if _models is None:
        if not FaceModels.available():
            raise EnrolError("face models missing: run setup (models/face/)")
        _models = FaceModels()
    return _models


async def load_gallery() -> Gallery:
    async with get_session() as s:
        rows = (await s.execute(select(FaceSample, KnownPerson.name)
                                .join(KnownPerson, KnownPerson.id == FaceSample.person_id))).all()
    if not rows:
        return Gallery()
    return Gallery(
        person_ids=np.array([r.FaceSample.person_id for r in rows], np.int64),
        names={r.FaceSample.person_id: r.name for r in rows},
        embeddings=np.stack([np.frombuffer(r.FaceSample.embedding, np.float32) for r in rows]))


async def list_people() -> list[dict]:
    async with get_session() as s:
        people = (await s.execute(select(KnownPerson).options(selectinload(KnownPerson.samples))
                                  .order_by(KnownPerson.name))).scalars().all()
        return [{"id": p.id, "name": p.name, "created_at": p.created_at,
                 "samples": [{"id": f.id, "quality": round(f.quality, 2)} for f in p.samples]}
                for p in people]


async def create_person(name: str) -> dict:
    name = " ".join(name.split())[:64]
    if not name:
        raise EnrolError("a name is required")
    async with get_session() as s:
        if (await s.execute(select(KnownPerson).where(KnownPerson.name == name))).scalar_one_or_none():
            raise EnrolError(f"{name} is already enrolled")
        p = KnownPerson(name=name)
        s.add(p)
        await s.commit()
        return {"id": p.id, "name": p.name, "samples": []}


async def add_sample(person_id: int, image_bytes: bytes) -> dict:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise EnrolError("not an image")
    m = models()
    faces = m.faces(img)
    if len(faces) > 1:
        sizes = sorted((f[2] * f[3] for f in faces), reverse=True)
        if sizes[1] > 0.5 * sizes[0]:
            raise EnrolError("several faces in this photo: use one showing only this person")
    f, why = best_face(m, img, float(_cfg("min_quality", 0.85)), ENROL_MIN_PX)
    if f is None:
        raise EnrolError(why)
    e, aligned = m.embed(img, f)
    gallery = await load_gallery()
    who, sim = gallery.match(e, float(_cfg("threshold", 0.363)), 0.0)
    async with get_session() as s:
        person = await s.get(KnownPerson, person_id)
        if person is None:
            raise EnrolError("no such person")
        if who is not None and who != person_id:
            raise EnrolError(f"this face matches {gallery.names[who]} ({sim:.2f}), "
                             "already enrolled: not added")
        # A person's photos must agree with each other: a wrong photo under a
        # name would make the system name the wrong face.
        own = gallery.embeddings[gallery.person_ids == person_id]
        if len(own):
            best_own = float((own @ e).max())
            if best_own < float(_cfg("threshold", 0.363)):
                raise EnrolError(f"this face does not look like the other photos of "
                                 f"{person.name} ({best_own:.2f}): not added")
        sample = FaceSample(person_id=person_id, embedding=e.tobytes(), image="",
                            quality=float(f[-1]))
        s.add(sample)
        await s.flush()
        rel = f"{person_id}/{sample.id}.jpg"
        (faces_dir() / str(person_id)).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(faces_dir() / rel), aligned, [cv2.IMWRITE_JPEG_QUALITY, 90])
        sample.image = rel
        await s.commit()
        return {"id": sample.id, "quality": round(sample.quality, 2)}


async def delete_sample(person_id: int, sample_id: int) -> None:
    async with get_session() as s:
        f = await s.get(FaceSample, sample_id)
        if f is None or f.person_id != person_id:
            raise EnrolError("no such photo")
        (faces_dir() / f.image).unlink(missing_ok=True)
        await s.delete(f)
        await s.commit()


async def delete_person(person_id: int) -> None:
    """Everything about this person: samples, images and the entry itself."""
    async with get_session() as s:
        p = (await s.execute(select(KnownPerson).options(selectinload(KnownPerson.samples))
                             .where(KnownPerson.id == person_id))).scalar_one_or_none()
        if p is None:
            raise EnrolError("no such person")
        await s.delete(p)
        await s.commit()
    shutil.rmtree(faces_dir() / str(person_id), ignore_errors=True)


async def sample_image(person_id: int, sample_id: int):
    async with get_session() as s:
        f = await s.get(FaceSample, sample_id)
    if f is None or f.person_id != person_id:
        return None
    path = (faces_dir() / f.image).resolve()
    return path if faces_dir().resolve() in path.parents and path.is_file() else None
