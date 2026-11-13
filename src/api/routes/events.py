"""Recent-events endpoint (paginated)."""
from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy import select

from src.storage import ActionRecord, Person, get_session

router = APIRouter()


@router.get("")
async def recent_events(
    limit: int = Query(50, le=500),
    label: str | None = None,
) -> list[dict]:
    async with get_session() as s:
        stmt = (
            select(ActionRecord, Person.external_id)
            .join(Person, Person.id == ActionRecord.person_id)
            .order_by(ActionRecord.start_time.desc())
            .limit(limit)
        )
        if label:
            stmt = stmt.where(ActionRecord.label == label)
        rows = (await s.execute(stmt)).all()
        return [
            {
                "person_id": ext,
                "label": a.label,
                "start_time": a.start_time.isoformat(),
                "end_time": a.end_time.isoformat(),
                "duration_s": a.duration_s,
                "confidence": a.confidence,
            }
            for a, ext in rows
        ]
