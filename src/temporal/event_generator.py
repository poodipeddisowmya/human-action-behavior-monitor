"""Turns smoothed per-frame labels into discrete, deduplicated action events.

An event is opened when a person's label changes and closed when it changes
again (or the person disappears). Only closed events are persisted, which
keeps the log meaningful and small.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from .smoother import SmoothedPrediction


@dataclass
class ActionEvent:
    person_id: int
    label: str
    start_time: datetime
    end_time: datetime | None = None
    confidences: list[float] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    @property
    def confidence(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "label": self.label,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_s": self.duration_s,
            "confidence": self.confidence,
        }


class EventGenerator:
    """Emits `ActionEvent` on close via the `on_event` callback."""

    def __init__(self, on_event: Callable[[ActionEvent], None]):
        self._on_event = on_event
        self._open: dict[int, ActionEvent] = {}

    def update(self, person_id: int, pred: SmoothedPrediction, now: datetime | None = None) -> None:
        now = now or datetime.utcnow()
        current = self._open.get(person_id)

        if current is None:
            self._open[person_id] = ActionEvent(
                person_id=person_id, label=pred.label, start_time=now,
                confidences=[pred.confidence],
            )
            return

        if current.label == pred.label:
            current.confidences.append(pred.confidence)
            return

        current.end_time = now
        self._on_event(current)
        self._open[person_id] = ActionEvent(
            person_id=person_id, label=pred.label, start_time=now,
            confidences=[pred.confidence],
        )

    def close_person(self, person_id: int, now: datetime | None = None) -> None:
        """Called when a track disappears (person leaves the frame)."""
        current = self._open.pop(person_id, None)
        if current:
            current.end_time = now or datetime.utcnow()
            self._on_event(current)

    def flush(self, now: datetime | None = None) -> None:
        for pid in list(self._open):
            self.close_person(pid, now)
