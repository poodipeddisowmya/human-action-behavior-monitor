"""Start the day with an empty activity log (before a demo).

Backs up the database to logs/backup/, then clears the activity log and the
list of tracked people. Enrolled people and their face photos are kept.
Proof photos already on disk stay until the retention period removes them.

Stop the monitor first: this refuses to run while it is answering.

    python scripts/reset_demo_log.py
"""
from __future__ import annotations

import sqlite3
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402


def _running() -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{settings.api_port}/api/health", timeout=2)
        return True
    except Exception:
        return False


def main() -> int:
    url = settings.database_url
    if not url.startswith("sqlite"):
        print("Only the default SQLite database is handled here; clear the 'actions' and "
              "'persons' tables of your database with its own tools.")
        return 1
    db = Path(url.split(":///", 1)[1])
    if not db.is_file():
        print(f"No database yet at {db}: nothing to reset.")
        return 0
    if _running():
        print(f"The monitor is running (port {settings.api_port}). Stop it, then run this again.")
        return 1

    backup_dir = db.parent / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{db.stem}-{datetime.now():%Y%m%d-%H%M%S}.db"
    con = sqlite3.connect(db)
    with sqlite3.connect(backup) as b:
        con.backup(b)
    events = con.execute("select count(*) from actions").fetchone()[0]
    people = con.execute("select count(*) from persons").fetchone()[0]
    con.execute("delete from actions")
    con.execute("delete from persons")
    con.commit()
    con.execute("vacuum")
    known = con.execute("select count(*) from known_people").fetchone()[0]
    con.close()
    print(f"Backed up to {backup.relative_to(ROOT) if ROOT in backup.parents else backup}")
    print(f"Cleared {events} logged activities and {people} tracked IDs; "
          f"kept {known} enrolled {'person' if known == 1 else 'people'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
