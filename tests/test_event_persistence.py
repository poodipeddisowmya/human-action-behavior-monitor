"""Concurrent events for one person must not lose events.

Regression test for a real failure: events are submitted from the pipeline
thread via run_coroutine_threadsafe, so two coroutines for the same person
raced on SELECT-then-INSERT. The loser hit
`UNIQUE constraint failed: persons.external_id`, dropped its event, and dumped
a full traceback. It reproduced within a minute of a fresh start.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import func, select

from src.api.runtime import Runtime
from src.storage import ActionRecord, Person, get_session, init_db, reset_engine
from src.temporal.event_generator import ActionEvent


# pytest_asyncio.fixture, not pytest.fixture: a plain @pytest.fixture on an
# async generator is handed to the test un-awaited, so the body never runs and
# the test silently uses the real database.
@pytest_asyncio.fixture
async def fresh_db(tmp_path, monkeypatch):
    """An empty database per test, isolated from logs/monitor.db."""
    from config import settings
    monkeypatch.setattr(
        settings, "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}",
    )
    reset_engine()
    await init_db()
    yield
    # Rebind to the real URL so a later test in this session is unaffected.
    monkeypatch.undo()
    reset_engine()


def _event(person_id: int, label: str, offset: int) -> ActionEvent:
    start = datetime.utcnow() + timedelta(seconds=offset)
    # duration_s and confidence are derived properties, not fields.
    return ActionEvent(
        person_id=person_id,
        label=label,
        start_time=start,
        end_time=start + timedelta(seconds=1),
        confidences=[0.9],
    )


@pytest.mark.asyncio
async def test_concurrent_events_same_person_all_persist(fresh_db):
    """Ten simultaneous events for one person: one row, ten actions, no loss."""
    rt = Runtime()
    events = [_event(7, f"label{i}", i) for i in range(10)]

    await asyncio.gather(*(rt._persist_and_publish(e) for e in events))

    async with get_session() as s:
        people = (await s.execute(
            select(Person).where(Person.external_id == "P007")
        )).scalars().all()
        n_actions = (await s.execute(
            select(func.count()).select_from(ActionRecord)
        )).scalar_one()

    assert len(people) == 1, f"expected one person row, got {len(people)}"
    assert n_actions == 10, f"lost {10 - n_actions} of 10 events to the race"


@pytest.mark.asyncio
async def test_concurrent_events_distinct_people(fresh_db):
    rt = Runtime()
    events = [_event(i, "walking", i) for i in range(8)]

    await asyncio.gather(*(rt._persist_and_publish(e) for e in events))

    async with get_session() as s:
        n_people = (await s.execute(
            select(func.count()).select_from(Person)
        )).scalar_one()
        n_actions = (await s.execute(
            select(func.count()).select_from(ActionRecord)
        )).scalar_one()

    assert n_people == 8
    assert n_actions == 8


@pytest.mark.asyncio
async def test_database_view_shows_rows_as_stored_with_their_camera(fresh_db):
    """The Database tab: newest row first, with the camera it was seen on."""
    from src.api.routes.database import database
    rt = Runtime()
    for i, (label, cam) in enumerate([("walking", "Reception"), ("using phone", "Pantry")]):
        e = _event(3, label, i)
        e.source = cam
        await rt._persist_and_publish(e)

    out = await database(limit=10)
    assert out["tables"]["actions"] == 2 and out["tables"]["persons"] == 1
    assert [r["label"] for r in out["rows"]] == ["using phone", "walking"]
    assert out["rows"][0]["source"] == "Pantry" and out["rows"][0]["person_id"] == "P003"
    assert out["database"]["engine"] == "SQLite" and out["database"]["bytes"] > 0
    assert out["written_last_minute"] == 2
