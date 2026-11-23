"""Proof images for logged activities (src/storage/evidence.py)."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import cv2
import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy import create_engine, text

from src.storage import evidence
from src.storage.evidence import EvidenceCollector, pick, render_shot

FRAME = np.full((540, 960, 3), 90, np.uint8)
BOX = (400, 100, 560, 500)


def test_render_shot_is_an_annotated_jpeg():
    jpg = render_shot(FRAME, BOX, "P007", "making tea", datetime(2026, 11, 18, 9, 30), 0.82)
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[0] > 100 and max(img.shape[:2]) <= 640 + 40


def test_pick_first_middle_last():
    shots = [(float(i), bytes([i])) for i in range(9)]
    assert pick(shots) == [bytes([0]), bytes([4]), bytes([8])]
    assert pick(shots[:2]) == [bytes([0]), bytes([1])]


def test_collector_keeps_shots_per_label_and_hands_them_over_once():
    c = EvidenceCollector(interval_s=0.0, enabled=True)
    for _ in range(5):
        c.observe(3, "drinking", 0.7, FRAME, BOX)
    assert len(c.take(3, "eating")) == 0           # not the label held
    shots = c.take(3, "drinking")
    assert len(shots) == 3
    assert c.take(3, "drinking") == []              # taken once


def test_collector_starts_afresh_when_the_label_changes():
    c = EvidenceCollector(interval_s=0.0, enabled=True)
    c.observe(1, "sitting", 0.7, FRAME, BOX)
    c.observe(1, "taking tablets", 0.6, FRAME, BOX)
    assert len(c.take(1, "taking tablets")) == 1
    assert c.take(1, "sitting") == []


def test_collector_skips_placeholder_labels_and_respects_interval():
    c = EvidenceCollector(interval_s=60.0, enabled=True)
    c.observe(2, "detecting…", 0.0, FRAME, BOX)
    assert c.take(2, "detecting…") == []
    for _ in range(4):
        c.observe(2, "reading", 0.6, FRAME, BOX)
    assert len(c.take(2, "reading")) == 1           # one shot per interval


def test_disabled_collector_takes_nothing():
    c = EvidenceCollector(interval_s=0.0, enabled=False)
    c.observe(1, "reading", 0.6, FRAME, BOX)
    assert c.take(1, "reading") == []


def test_save_and_purge(tmp_path, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    old = datetime(2026, 1, 1, 12)
    paths = evidence.save(5, [b"a", b"b"], old)
    assert paths == ["2026-01-01/5_0.jpg", "2026-01-01/5_1.jpg"]
    evidence.save(6, [b"c"], datetime(2026, 2, 10, 12))
    assert evidence.purge_old(30, today=date(2026, 2, 15)) == 1
    assert not (tmp_path / "evidence" / "2026-01-01").exists()
    assert (tmp_path / "evidence" / "2026-02-10" / "6_0.jpg").exists()
    assert evidence.purge_old(0, today=date(2030, 1, 1)) == 0    # 0 = keep forever


@pytest_asyncio.fixture
async def fresh_db(tmp_path, monkeypatch):
    from config import settings
    from src.storage import init_db, reset_engine
    monkeypatch.setattr(settings, "database_url",
                        f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    reset_engine()
    await init_db()
    yield tmp_path
    monkeypatch.undo()
    reset_engine()


@pytest.mark.asyncio
async def test_event_with_evidence_is_stored_and_served(fresh_db):
    from fastapi.testclient import TestClient
    from src.api.routes import evidence as route
    from src.api.runtime import Runtime
    from src.storage import ActionRecord, get_session
    from src.temporal.event_generator import ActionEvent
    from fastapi import FastAPI

    start = datetime.utcnow()
    jpg = render_shot(FRAME, BOX, "P009", "taking tablets", start)
    ev = ActionEvent(person_id=9, label="taking tablets", start_time=start,
                     end_time=start + timedelta(seconds=4), confidences=[0.7],
                     evidence=[jpg, jpg])
    await Runtime()._persist_and_publish(ev)

    async with get_session() as s:
        rec = (await s.execute(ActionRecord.__table__.select())).first()
    assert len(json.loads(rec.evidence)) == 2

    app = FastAPI()
    app.include_router(route.router, prefix="/api/evidence")
    client = TestClient(app)
    r = client.get(f"/api/evidence/{rec.id}/1")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/evidence/{rec.id}/2").status_code == 404
    assert client.get("/api/evidence/99999/0").status_code == 404


@pytest.mark.asyncio
async def test_old_database_gains_the_evidence_column(tmp_path, monkeypatch):
    """A monitor.db from before this feature keeps its history and gets the column."""
    from config import settings
    from src.storage import init_db, reset_engine
    db = tmp_path / "old.db"
    eng = create_engine(f"sqlite:///{db.as_posix()}")
    with eng.begin() as c:
        c.execute(text("CREATE TABLE persons (id INTEGER PRIMARY KEY, external_id VARCHAR(32), "
                       "first_seen DATETIME, last_seen DATETIME)"))
        c.execute(text("CREATE TABLE actions (id INTEGER PRIMARY KEY, person_id INTEGER, "
                       "label VARCHAR(32), start_time DATETIME, end_time DATETIME, "
                       "duration_s FLOAT, confidence FLOAT)"))
        c.execute(text("INSERT INTO actions (person_id, label, duration_s, confidence) "
                       "VALUES (1, 'reading', 3.0, 0.6)"))
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{db.as_posix()}")
    reset_engine()
    await init_db()
    with eng.begin() as c:
        cols = [r[1] for r in c.execute(text("PRAGMA table_info(actions)"))]
        assert "evidence" in cols
        assert c.execute(text("SELECT label FROM actions")).scalar() == "reading"
    monkeypatch.undo()
    reset_engine()
