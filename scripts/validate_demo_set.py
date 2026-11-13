"""Validate that each demo clip produces its expected action label.

Switches the running pipeline through every demo clip, watches the backend log
for VideoMAE predictions, and prints a pass/fail table.

Usage (backend must already be running):
    python scripts/validate_demo_set.py --log <path-to-backend-log>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

API = "http://localhost:8000"

# clip filename -> expected category (None = pose-only, no deep label expected)
EXPECTED: dict[str, str | None] = {
    "demo-eating.avi":            "eating",
    "demo-drinking.mp4":          "drinking",
    "demo-cooking.mp4":           "cooking",
    "demo-cleaning-mopping.avi":  "cleaning",
    "demo-using-computer.avi":    "using computer",
    "demo-writing.avi":           "writing",
    "demo-cycling.avi":           "cycling",
    "demo-exercising-pushup.avi": "exercising",
    "demo-exercising-squats.avi": "exercising",
    "demo-exercising-pullup.avi": "exercising",
    "demo-basic-sit.avi":         None,
    "demo-basic-walk.avi":        None,
    "demo-basic-run.avi":         None,
}

# Parse the STRUCTURED line the classifier emits, not the human-readable one.
# Format:  DEEPPRED \t person_id \t category \t confidence \t coverage
# Scraping the prose log broke this harness three times during development.
PRED_RE = re.compile(r"^DEEPPRED\t(\d+)\t([^\t]+)\t([0-9.]+)\t([0-9.]+)")


def _parse(line: str):
    m = PRED_RE.match(line)
    return (m.group(2), float(m.group(3))) if m else None


def switch(filename: str) -> None:
    req = urllib.request.Request(
        f"{API}/api/sources",
        data=json.dumps({"filename": filename}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=10).read()


def recent_labels(log: Path, from_line: int) -> Counter:
    text = log.read_text(errors="ignore").splitlines()
    c = Counter()
    for line in text[from_line:]:
        parsed = _parse(line)
        if parsed:
            c[parsed[0]] += 1
    return c


def line_count(log: Path) -> int:
    try:
        return len(log.read_text(errors="ignore").splitlines())
    except FileNotFoundError:
        return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="backend stdout log path")
    ap.add_argument("--dwell", type=float, default=22.0, help="seconds per clip")
    args = ap.parse_args()
    log = Path(args.log)

    samples = Path(__file__).resolve().parent.parent / "data" / "samples"
    rows, passes, total = [], 0, 0

    for clip, expected in EXPECTED.items():
        if not (samples / clip).exists():
            rows.append((clip, expected, "clip missing", "SKIP"))
            continue

        start = line_count(log)
        switch(clip)
        time.sleep(args.dwell)
        counts = recent_labels(log, start)

        if expected is None:
            got = ", ".join(f"{k}({v})" for k, v in counts.most_common(2)) or "no deep label"
            rows.append((clip, "pose only", got, "—"))
            continue

        total += 1
        top = counts.most_common(1)[0][0] if counts else None
        got = ", ".join(f"{k}({v})" for k, v in counts.most_common(2)) or "— none —"
        ok = top == expected
        if ok:
            passes += 1
        rows.append((clip, expected, got, "PASS" if ok else "FAIL"))

    w1 = max(len(r[0]) for r in rows) + 2
    w2 = max(len(str(r[1])) for r in rows) + 2
    print()
    print(f"{'CLIP':<{w1}}{'EXPECTED':<{w2}}{'GOT':<34}RESULT")
    print("-" * (w1 + w2 + 42))
    for clip, exp, got, res in rows:
        print(f"{clip:<{w1}}{str(exp):<{w2}}{got:<34}{res}")
    print()
    if total:
        print(f"Deep-label accuracy: {passes}/{total} ({100*passes//total}%)")


if __name__ == "__main__":
    main()
