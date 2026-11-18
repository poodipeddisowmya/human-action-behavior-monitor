"""Shared runtime state between the pipeline thread and the async API.

Holds the latest annotated frame (for MJPEG streaming) and offers helpers to
persist events + push them onto the websocket broadcaster.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from threading import Lock
from typing import Optional

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from config import settings
from src.storage import ActionRecord, Person, evidence, get_session
from src.storage.evidence import evidence_urls
from src.temporal.event_generator import ActionEvent
from src.utils import get_logger

from .websocket import broadcaster

log = get_logger(__name__)


class Runtime:
    def __init__(self):
        self._frame_lock = Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_annotations: list[dict] = []
        self._latest_caption: dict | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.source = None  # set by the lifespan to a SwappableFileSource
        self.pipeline = None  # set by the lifespan; read for /api/metrics
        # Frame sequence number, and the JPEG of that frame once encoded.
        # Every viewer of /api/stream/video shares the one encode instead of
        # each connection re-encoding the same frame.
        self._seq = 0
        self._jpeg: tuple[int, bytes] | None = None
        self._jpeg_lock = Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def frame_seq(self) -> int:
        return self._seq

    def latest_jpeg(self) -> tuple[int, bytes] | None:
        """(seq, jpeg bytes) of the newest frame, encoded at most once."""
        with self._frame_lock:
            seq, frame = self._seq, self._latest_frame
        if frame is None:
            return None
        with self._jpeg_lock:
            if self._jpeg is not None and self._jpeg[0] == seq:
                return self._jpeg
            h, w = frame.shape[:2]
            max_w = settings.stream_max_width
            if max_w and w > max_w:
                frame = cv2.resize(frame, (max_w, int(h * max_w / w)),
                                   interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, settings.stream_jpeg_quality])
            if not ok:
                return self._jpeg
            self._jpeg = (seq, buf.tobytes())
            return self._jpeg

    # ---- frame side ----
    @property
    def latest_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    @property
    def latest_annotations(self) -> list[dict] | None:
        with self._frame_lock:
            return list(self._latest_annotations) if self._latest_annotations else None

    @property
    def latest_caption(self) -> dict | None:
        with self._frame_lock:
            return dict(self._latest_caption) if self._latest_caption else None

    def on_frame(self, frame: np.ndarray, annotations: list[dict],
                 caption: dict | None = None) -> None:
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_annotations = annotations
            self._latest_caption = caption
            self._seq += 1
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                broadcaster.publish, {"type": "frame_annotations",
                                      "annotations": annotations,
                                      "caption": caption}
            )

    # ---- event side ----
    def on_event(self, event: ActionEvent) -> None:
        if self._loop is None:
            return
        # "detecting…" is the placeholder before a person's first label, not
        # an activity: logging it filled the events table with noise.
        if event.label.startswith("detecting"):
            return
        asyncio.run_coroutine_threadsafe(self._persist_and_publish(event), self._loop)

    async def _persist_and_publish(self, event: ActionEvent) -> None:
        try:
            async with get_session() as s:
                ext = f"P{event.person_id:03d}"
                person = await self._get_or_create_person(s, ext, event)
                person.last_seen = event.end_time or datetime.utcnow()

                record = ActionRecord(
                    person_id=person.id,
                    label=event.label,
                    start_time=event.start_time,
                    end_time=event.end_time or datetime.utcnow(),
                    duration_s=event.duration_s,
                    confidence=event.confidence,
                )
                s.add(record)
                await s.flush()                      # assigns record.id
                if event.evidence:
                    try:
                        paths = await asyncio.to_thread(
                            evidence.save, record.id, event.evidence, event.start_time)
                        record.evidence = json.dumps(paths)
                    except OSError as e:             # disk full etc.: keep the event
                        log.warning(f"evidence not saved for event {record.id}: {e}")
                await s.commit()
            broadcaster.publish({"type": "event", **event.to_dict(),
                                 "id": record.id,
                                 "evidence": evidence_urls(record),
                                 "person_id": f"P{event.person_id:03d}"})
        except Exception as e:
            log.exception(f"failed to persist event: {e}")

    @staticmethod
    async def _get_or_create_person(s, ext: str, event: ActionEvent) -> Person:
        """Fetch the person row, creating it if this is the first sighting.

        SELECT-then-INSERT is a race here, not a theoretical one. Events are
        submitted from the pipeline thread with run_coroutine_threadsafe, so
        two coroutines for the same person can both miss the SELECT and both
        INSERT — the second then dies on the unique index, the event is lost,
        and a full traceback goes to the log. It fired within a minute of a
        fresh start on a looping clip.

        Letting the insert fail and re-reading is the fix: whichever writer
        lost the race simply adopts the row the winner created. Retrying the
        SELECT (rather than the INSERT) is what makes this terminate.
        """
        stmt = select(Person).where(Person.external_id == ext)
        person = (await s.execute(stmt)).scalar_one_or_none()
        if person is not None:
            return person

        person = Person(external_id=ext, first_seen=event.start_time,
                        last_seen=event.end_time or datetime.utcnow())
        s.add(person)
        try:
            await s.flush()
            return person
        except IntegrityError:
            # Someone else created it between our SELECT and our INSERT.
            await s.rollback()
            existing = (await s.execute(stmt)).scalar_one_or_none()
            if existing is None:
                raise       # a different constraint failed; do not swallow it
            return existing


runtime = Runtime()
