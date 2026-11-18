"""SQLAlchemy models for persistent activity logs."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
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

    person: Mapped["Person"] = relationship(back_populates="actions")
