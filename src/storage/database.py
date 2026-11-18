"""Async SQLAlchemy engine + session factory."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import settings

from .models import Base

_engine = create_async_engine(settings.database_url, echo=False, future=True)
_Session = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


def reset_engine() -> None:
    """Rebuild the engine from the current settings.database_url.

    The engine binds at import, which is right for the running app but means a
    test pointing DATABASE_URL at a temp file would still write to the real
    logs/monitor.db — passing or failing against whatever data happens to be
    sitting there. This gives tests a way to rebind. The application itself
    never calls it.
    """
    global _engine, _Session
    _engine = create_async_engine(settings.database_url, echo=False, future=True)
    _Session = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)


def _add_missing_columns(conn) -> None:
    """create_all makes missing tables, not missing columns: add the ones a
    database from an earlier version lacks, so it keeps its history."""
    insp = inspect(conn)
    for table in Base.metadata.sorted_tables:
        if not insp.has_table(table.name):
            continue
        have = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name not in have and col.nullable:
                ddl = col.type.compile(dialect=conn.dialect)
                conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {ddl}'))


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    async with _Session() as s:
        yield s
