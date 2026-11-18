"""Timestamps as the dashboard should read them."""
from __future__ import annotations

def utc_iso(dt) -> str | None:
    """ISO 8601 with a Z: the app stores naive UTC (datetime.utcnow), and a
    browser reads a timestamp without a zone as LOCAL time, so the dashboard
    showed UTC clock times as if they were local."""
    if dt is None:
        return None
    return dt.isoformat() + "Z" if dt.tzinfo is None else dt.isoformat()
