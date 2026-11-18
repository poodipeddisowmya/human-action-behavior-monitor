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
    """Pairs and totals over every clip recorded on both sides.

    Optional "extra_sides" (scripts/add_showcase_side.py) are further devices
    recorded on the same clips, e.g. a Jetson: each is summarised over the
    same labelled clips, and a clip it lacks shows as null.
    """
    left = {Path(r["video"]).stem: r for r in data.get("before", [])}
    right = {Path(r["video"]).stem: r for r in data.get("clips", [])}
    both = [c for c in right if c in left]
    # Recorded on the GPUs only (activities added after the CPU recording):
    # listed with no CPU side, and totalled separately.
    gpu_only = [c for c in right if c not in left]
    gpu_only_labelled = [c for c in gpu_only if right[c].get("expected")]
    labelled = [c for c in both if right[c].get("expected")]
    b, a = [left[c] for c in labelled], [right[c] for c in labelled]
    n = len(labelled)
    extras = [(x, {Path(r["video"]).stem: {**r, "expected": right[Path(r["video"]).stem].get("expected")}
                   for r in x.get("clips", []) if Path(r["video"]).stem in right})
              for x in data.get("extra_sides", [])]

    def totals(rows, n=n):
        # "appeared" counts clips where the correct label was shown at all; the
        # median time is over those only, so it must never be read alone (a
        # side that rarely gets there would otherwise look fast).
        return {"clear": sum(share(r) >= CLEAR_SHARE for r in rows),
                "appeared": sum(first_shown(r) is not None for r in rows),
                "mean_share": round(sum(map(share, rows)) / n, 3) if n else None,
                "median_first_s": _median(map(first_shown, rows)),
                "median_fps": _median(r.get("processing_fps") for r in rows)}

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
            "clips": len(gpu_only_labelled),
            "totals": {"right": totals([right[c] for c in gpu_only_labelled], len(gpu_only_labelled)),
                       **{x["key"]: totals([rows[c] for c in gpu_only_labelled if c in rows],
                                           len(gpu_only_labelled)) for x, rows in extras}},
        } if gpu_only_labelled else None,
        "clips": [{"name": c, "expected": right[c].get("expected"),
                   "featured": c in (data.get("compare") or []) or c in gpu_only_labelled,
                   "new_activity": c in gpu_only,
                   "left": side(left[c]) if c in left else None, "right": side(right[c]),
                   "extra": {x["key"]: side(rows[c]) if c in rows else None
                             for x, rows in extras}} for c in [*both, *gpu_only]],
    }
