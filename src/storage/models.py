"""SQLAlchemy models for persistent activity logs."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)  # e.g. "P001"
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    actions: Mapped[list["ActionRecord"]] = relationship(
        back_populates="person", cascade="all, delete-orphan"
    )


class ActionRecord(Base):
    __tablename__ = "actions"

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("persons.id"), index=True)
    label: Mapped[str] = mapped_column(String(32), index=True)
    start_time: Mapped[datetime] = mapped_column(DateTime, index=True)
    end_time: Mapped[datetime] = mapped_column(DateTime)
    duration_s: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    # JSON list of proof images, relative to data/evidence/ (see
    # src/storage/evidence.py); NULL when none were taken.
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    # The enrolled person this was, if face recognition named them (the name
    # at the time: it survives the person being deleted from the gallery).
    identity: Mapped[str | None] = mapped_column(String(64), nullable=True, default=None)
    # Where it was seen: the CCTV camera's name, "USB camera", or the video
    # file. NULL on records from before this column existed.
    source: Mapped[str | None] = mapped_column(String(128), nullable=True, default=None)

    person: Mapped["Person"] = relationship(back_populates="actions")


class KnownPerson(Base):
    """Someone enrolled for face recognition (src/identity/face_id.py).

    Only enrolled people are ever named; everyone else stays "Unknown".
    Enrol only with the person's consent: a face is biometric data.
    """
    __tablename__ = "known_people"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    samples: Mapped[list["FaceSample"]] = relationship(
        back_populates="person", cascade="all, delete-orphan")


class FaceSample(Base):
    """One enrolled face: its SFace embedding and the aligned face image."""
    __tablename__ = "face_samples"

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("known_people.id"), index=True)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)       # 128 float32
    image: Mapped[str] = mapped_column(String(128))             # under data/faces/
    quality: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    person: Mapped["KnownPerson"] = relationship(back_populates="samples")
