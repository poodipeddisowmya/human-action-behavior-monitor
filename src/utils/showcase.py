"""Shared summary of a CPU-vs-GPU showcase recording.

A showcase recording (kaggle/gpu_showcase.py + scripts/set_baseline.py) is a
showcase.json whose "clips" are one side (with GPU) and "before" the other
(without GPU), every clip played 12 s through the real pipeline. Both the
static showcase page (scripts/build_showcase.py) and the dashboard's
"CPU vs GPU" tab (/api/showcase) summarise it the same way, here.
"""
from __future__ import annotations

from pathlib import Path

CLEAR_SHARE = 0.5     # "clearly recognised": correct label on screen >= half the time


def share(row: dict) -> float:
    return (row.get("labels_shown") or {}).get(row.get("expected"), 0.0)


def first_shown(row: dict) -> float | None:
    return (row.get("first_seen_s") or {}).get(row.get("expected"))


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def side(row: dict) -> dict:
    """What one side showed for one clip."""
    return {"video": row.get("annotated_video"),
            "share": round(share(row), 3) if row.get("expected") else None,
            "first_s": first_shown(row),
            "fps": row.get("processing_fps"),
            "people": row.get("tracked_mean"),
            "top": dict(list((row.get("labels_shown") or {}).items())[:3])}


def summarize(data: dict) -> dict:
    """Pairs and totals over the comparison recording.

    "before" is the laptop CPU, "clips" the GPU (T4), and optional
    "extra_sides" further devices (scripts/add_showcase_side.py), e.g. the
    Jetson. The headline totals are over the labelled clips that the CPU and
    the GPU both recorded (each extra device over the same clips). Clips
    recorded on only some devices are still listed, with null for the
    others; activities added after the CPU recording (no CPU side) are
    totalled separately as "new activities".
    """
    left = {Path(r["video"]).stem: r for r in data.get("before", [])}
    right = {Path(r["video"]).stem: r for r in data.get("clips", [])}
    extras = [(x, {Path(r["video"]).stem: r for r in x.get("clips", [])})
              for x in data.get("extra_sides", [])]

    def expected(c: str):
        """The GPU recording's label for a clip, else any device's."""
        for rows in (right, *[rows for _, rows in extras], left):
            if c in rows and rows[c].get("expected"):
                return rows[c]["expected"]
        return None

    # Every expected label comes from the same place, whichever side is read.
    extras = [(x, {c: {**r, "expected": expected(c)} for c, r in rows.items()})
              for x, rows in extras]
    both = [c for c in right if c in left]
    labelled = [c for c in both if expected(c)]
    b, a = [left[c] for c in labelled], [right[c] for c in labelled]
    n = len(labelled)
    order = list(dict.fromkeys([*both, *right, *[c for _, rows in extras for c in rows]]))
    new_clips = [c for c in order if c not in left and expected(c)]

    def totals(rows, n=n):
        # "appeared" counts clips where the correct label was shown at all; the
        # median time is over those only, so it must never be read alone (a
        # side that rarely gets there would otherwise look fast).
        return {"clear": sum(share(r) >= CLEAR_SHARE for r in rows),
                "appeared": sum(first_shown(r) is not None for r in rows),
                "mean_share": round(sum(map(share, rows)) / n, 3) if n else None,
                "median_first_s": _median(map(first_shown, rows)),
                "median_fps": _median(r.get("processing_fps") for r in rows)}

    def new_totals(rows: dict):
        got = [rows[c] for c in new_clips if c in rows]
        return totals(got, len(got)) | {"recorded": len(got)} if got else None

    return {
        "left": data.get("compare_left") or {"tag": "WITHOUT GPU", "sub": "CPU"},
        "right": data.get("compare_right") or {"tag": "WITH GPU", "sub": data.get("gpu") or "GPU"},
        "recorded": data.get("recorded"),
        "seconds_per_clip": data.get("seconds_per_clip"),
        "labelled_clips": n,
        "totals": {"left": totals(b), "right": totals(a)},
        "extra": [{"key": x["key"], "tag": x.get("tag") or x["key"].upper(),
                   "sub": x.get("sub") or "", "short": x.get("short") or x["key"],
                   "recorded": x.get("recorded"),
                   "totals": totals([rows[c] for c in labelled if c in rows])}
                  for x, rows in extras],
        "new_activities": {
            "clips": len(new_clips),
            "labels": sorted({expected(c) for c in new_clips}),
            "totals": {k: t for k, t in (("right", new_totals(right)),
                                         *[(x["key"], new_totals(rows)) for x, rows in extras])
                       if t},
        } if new_clips else None,
        "clips": [{"name": c, "expected": expected(c),
                   "featured": c in (data.get("compare") or []) or c in new_clips,
                   "new_activity": c in new_clips,
                   "left": side(left[c]) if c in left else None,
                   "right": side(right[c]) if c in right else None,
                   "extra": {x["key"]: side(rows[c]) if c in rows else None
                             for x, rows in extras}} for c in order],
    }
