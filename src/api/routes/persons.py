"""REST endpoints for people and their activity histories."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from src.storage import ActionRecord, Person, get_session

router = APIRouter()


@router.get("")
async def list_persons() -> list[dict]:
    async with get_session() as s:
        rows = (await s.execute(select(Person).order_by(Person.last_seen.desc()))).scalars().all()
        return [
            {
                "id": p.external_id,
                "first_seen": p.first_seen.isoformat(),
                "last_seen": p.last_seen.isoformat(),
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
            "first_seen": person.first_seen.isoformat(),
            "last_seen": person.last_seen.isoformat(),
            "history": [
                {
                    "label": a.label,
                    "start_time": a.start_time.isoformat(),
                    "end_time": a.end_time.isoformat(),
                    "duration_s": a.duration_s,
                    "confidence": a.confidence,
                }
                for a in actions
            ],
        }
