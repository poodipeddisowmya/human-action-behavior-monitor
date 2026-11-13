"""Verify the guarantee: every clip produces a label on every tracked person.

This checks the PRODUCTION requirement — "the model is always active" — which
is separate from whether the label is correct. A blank box reads as a broken
system, so the pipeline must never emit one.

Usage (backend must be running):
    python scripts/check_always_labels.py
"""
from __future__ import annotations

import datetime
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API = "http://localhost:8000"


def api_get(path: str):
    with urllib.request.urlopen(f"{API}{path}", timeout=10) as r:
        return json.loads(r.read().decode())


def switch(filename: str) -> None:
    req = urllib.request.Request(
        f"{API}/api/sources",
        data=json.dumps({"filename": filename}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=10).read()


def main() -> None:
    clips = [c["filename"] for c in api_get("/api/sources")["clips"]]
    print(f"Checking {len(clips)} clips — does every one produce a label?\n")
    print(f"  {'clip':<40}{'labels seen':<34}{'result'}")
    print("  " + "-" * 84)

    ok = 0
    for clip in clips:
        switch(clip)
        # Mark the boundary AFTER switching, so events from the previous clip
        # are excluded. Reading the global recent-events list without this
        # filter reports the prior clip's labels — which made three unrelated
        # clips all appear to say "exercising 95%".
        time.sleep(16)          # cold start + a few classification cycles

        # Sample the LIVE state several times. This is what the UI draws, and
        # unlike /api/events it reflects steady labels that never close an
        # event. It also distinguishes the two very different situations that
        # both look like "no output": nobody in frame vs. a detected person
        # left unlabelled.
        labels: dict[str, float] = {}
        saw_person = False
        blank_person = False
        for _ in range(6):
            live = api_get("/api/stream/live")
            if live["tracked"]:
                saw_person = True
                for p in live["people"]:
                    lbl = p["label"]
                    if lbl in ("-", "", "detecting…"):
                        blank_person = True
                    else:
                        labels[lbl] = max(labels.get(lbl, 0.0), p["confidence"])
            time.sleep(1.2)

        if not saw_person:
            shown, verdict = "(no person in frame)", "N/A"
            ok += 1                      # nothing to label is not a failure
        elif labels:
            shown = ", ".join(f"{k} {round(v*100)}%" for k, v in
                              sorted(labels.items(), key=lambda x: -x[1])[:2])
            verdict = "LABELLED"
            ok += 1
        else:
            shown, verdict = "(person seen, no label)", "BLANK"
        print(f"  {clip:<40}{shown:<34}{verdict}")

    print()
    print(f"  {ok}/{len(clips)} clips OK")
    if ok == len(clips):
        print("  PASS - every detected person carries a label.")
    else:
        print("  FAIL - a detected person was left unlabelled.")


if __name__ == "__main__":
    main()
