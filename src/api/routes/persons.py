"""REST endpoints for people and their activity histories."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from src.storage import ActionRecord, Person, get_session
from src.storage.evidence import evidence_urls
from src.utils.timefmt import utc_iso

router = APIRouter()


@router.get("")
async def list_persons() -> list[dict]:
    async with get_session() as s:
        rows = (await s.execute(select(Person).order_by(Person.last_seen.desc()))).scalars().all()
        return [
            {
                "id": p.external_id,
                "first_seen": utc_iso(p.first_seen),
                "last_seen": utc_iso(p.last_seen),
            }
            for p in rows
        ]


@router.get("/{external_id}")
async def get_person(external_id: str) -> dict:
    async with get_session() as s:
        person = (
            await s.execute(select(Person).where(Person.external_id == external_id))
        ).scalar_one_or_none()
        if not person:
            raise HTTPException(404, "person not found")

        actions = (
            await s.execute(
                select(ActionRecord)
                .where(ActionRecord.person_id == person.id)
                .order_by(ActionRecord.start_time.desc())
                .limit(200)
            )
        ).scalars().all()

        return {
            "id": person.external_id,
            "first_seen": utc_iso(person.first_seen),
            "last_seen": utc_iso(person.last_seen),
            "history": [
                {
                    "id": a.id,
                    "label": a.label,
                    "start_time": utc_iso(a.start_time),
                    "end_time": utc_iso(a.end_time),
                    "duration_s": a.duration_s,
                    "confidence": a.confidence,
                    "evidence": evidence_urls(a),
                    "identity": a.identity,
                }
                for a in actions
            ],
        }
