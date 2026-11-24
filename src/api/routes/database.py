"""What is stored, as it is stored: the dashboard's Database tab.

Rows of the `actions` table newest first (by insertion order, i.e. id), the
size of each table, where the database and the proof photos live on disk,
and how fast rows are being written. Read-only.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from config import settings
from src.storage import ActionRecord, FaceSample, KnownPerson, Person, get_session
from src.storage.evidence import evidence_root, evidence_urls
from src.utils.timefmt import utc_iso

router = APIRouter()

# Walking data/evidence/ costs a few ms per thousand photos: once per 15 s
# is plenty for a dashboard that polls every few seconds.
_EVIDENCE_TTL_S = 15.0
_evidence_cache: tuple[float, dict] | None = None


def _evidence_on_disk() -> dict:
    global _evidence_cache
    if _evidence_cache and time.monotonic() - _evidence_cache[0] < _EVIDENCE_TTL_S:
        return _evidence_cache[1]
    files = size = 0
    root = evidence_root()
    if root.is_dir():
        for dirpath, _, names in os.walk(root):
            for n in names:
                if n.endswith(".jpg"):
                    files += 1
                    size += os.path.getsize(os.path.join(dirpath, n))
    out = {"folder": _shown(root), "photos": files, "bytes": size,
           "retention_days": settings.evidence_retention_days}
    _evidence_cache = (time.monotonic(), out)
    return out


def _shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(settings.data_dir.parent.resolve()))
    except ValueError:
        return str(path)


def _database_file() -> dict:
    url = settings.database_url
    if not url.startswith("sqlite"):
        return {"engine": url.split(":", 1)[0].split("+", 1)[0], "file": None, "bytes": None}
    db = Path(url.split(":///", 1)[1])
    size = sum(p.stat().st_size for p in (db, Path(f"{db}-wal")) if p.exists())
    return {"engine": "SQLite", "file": _shown(db), "bytes": size}


def row(a: ActionRecord, ext: str) -> dict:
    """One `actions` row as stored, plus its photo links."""
    return {
        "id": a.id,
        "person_id": ext,
        "identity": a.identity,
        "label": a.label,
        "start_time": utc_iso(a.start_time),
        "end_time": utc_iso(a.end_time),
        "duration_s": round(a.duration_s, 2),
        "confidence": round(a.confidence, 3),
        "source": a.source,
        "evidence": evidence_urls(a),
    }


@router.get("")
async def database(limit: int = Query(25, ge=1, le=200)) -> dict:
    now = datetime.utcnow()
    async with get_session() as s:
        async def count(model, *where):
            return (await s.execute(select(func.count()).select_from(model).where(*where))).scalar_one()

        tables = {
            "actions": await count(ActionRecord),
            "persons": await count(Person),
            "known_people": await count(KnownPerson),
            "face_samples": await count(FaceSample),
        }
        named = await count(ActionRecord, ActionRecord.identity.is_not(None))
        last_minute = await count(ActionRecord, ActionRecord.end_time >= now - timedelta(minutes=1))
        last_hour = await count(ActionRecord, ActionRecord.end_time >= now - timedelta(hours=1))
        rows = (await s.execute(
            select(ActionRecord, Person.external_id)
            .join(Person, Person.id == ActionRecord.person_id)
            .order_by(ActionRecord.id.desc()).limit(limit))).all()
    return {
        "database": _database_file(),
        "tables": tables,
        "named_actions": named,
        "written_last_minute": last_minute,
        "written_last_hour": last_hour,
        "last_write": rows[0][0].end_time.isoformat() + "Z" if rows else None,
        "evidence": _evidence_on_disk(),
        "rows": [row(a, ext) for a, ext in rows],
    }
