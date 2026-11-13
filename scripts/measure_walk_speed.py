"""Measure the actual motion signal on clips, to calibrate _STAND_MAX.

The pose classifier calls anything under _STAND_MAX body-heights/sec
"standing". Slow walking in portrait phone video may never cross it, because
the bounding box barely moves relative to a tall body. This prints the real
distribution so the threshold is set from data rather than guessed.

    python scripts/measure_walk_speed.py walking.mp4 standing.mp4
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.action_recognition import action_classifier as AC  # noqa: E402
from src.tracking import PersonTracker  # noqa: E402


def measure_baseline(clip: Path, max_frames: int, lag: int) -> list[float]:
    """Net displacement over `lag` frames, normalised by body height.

    Frame-to-frame deltas at walking speed (~0.004 body-heights/frame) are
    below ByteTrack's box jitter, so the per-frame signal is mostly noise.
    Over a longer baseline real translation grows linearly while jitter grows
    as sqrt(t), so the ratio improves with lag.
    """
    tracker = PersonTracker()
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    hist: dict[int, list[tuple[float, float, float]]] = {}
    out: list[float] = []
    n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        for t in tracker.update(frame):
            x1, y1, x2, y2 = t.xyxy
            h = hist.setdefault(t.person_id, [])
            h.append(((x1 + x2) / 2, (y1 + y2) / 2, max(1.0, y2 - y1)))
            if len(h) > lag:
                px, py, pbh = h[-lag - 1]
                cx, cy, bh = h[-1]
                norm = (pbh + bh) / 2
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                out.append(dist / norm * (fps / lag))
    cap.release()
    return out


def measure(clip: Path, max_frames: int) -> list[float]:
    tracker = PersonTracker()
    pose = AC.SkeletonHeuristicClassifier()
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    speeds: list[float] = []
    prev: dict[int, tuple[float, float, float]] = {}
    n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        for t in tracker.update(frame):
            x1, y1, x2, y2 = t.xyxy
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            body_h = max(1.0, y2 - y1)
            if t.person_id in prev:
                px, py, pbh = prev[t.person_id]
                # Normalise by body height so distance-to-camera cancels out.
                norm = (pbh + body_h) / 2
                dx, dy = abs(cx - px) / norm * fps, (cy - py) / norm * fps
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 / norm * fps
                # Sitting down shrinks the box vertically; walking preserves it.
                speeds.append((dist, dx, dy, body_h / pbh))
            prev[t.person_id] = (cx, cy, body_h)
            pose.update(t.person_id, t.keypoints, t.xyxy, None)
    cap.release()
    return speeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--lags", type=int, nargs="+", default=[1, 6, 12, 25])
    args = ap.parse_args()

    # Does a longer measurement baseline separate walking from sitting-down?
    print("\n  median speed (body-heights/sec) at increasing baselines:\n")
    print(f"  {'clip':<40} " + " ".join(f"{'lag'+str(l):>8}" for l in args.lags))
    for name in args.clips:
        path = Path("data/samples") / name
        if not path.exists():
            continue
        cells = []
        for lag in args.lags:
            vals = measure_baseline(path, args.frames, lag)
            cells.append(f"{statistics.median(vals):>8.3f}" if vals else f"{'--':>8}")
        print(f"  {name[:40]:<40} " + " ".join(cells))
    print()

    print(f"\n  current thresholds: STAND<{AC._STAND_MAX}  WALK<{AC._WALK_MAX}  else run")
    print("  |dx| = horizontal speed, dy = vertical (+ = moving down)")
    print("  hgt  = bbox height ratio vs previous frame (<1 = shrinking)\n")
    print(f"  {'clip':<40} {'n':>4} {'spd':>6} {'|dx|':>6} {'dy':>7} {'hgt':>6} {'dx>dy':>6}")
    for name in args.clips:
        path = Path("data/samples") / name
        if not path.exists():
            print(f"  {name[:40]:<40} -- not found")
            continue
        rows = measure(path, args.frames)
        if not rows:
            print(f"  {name[:40]:<40} -- no tracks")
            continue
        med = lambda i: statistics.median([r[i] for r in rows])  # noqa: E731
        # Fraction of frames where motion is mostly horizontal — the thing
        # that should tell walking apart from lowering yourself into a chair.
        horiz = sum(1 for r in rows if r[1] > abs(r[2])) / len(rows)
        print(
            f"  {name[:40]:<40} {len(rows):>4} {med(0):>6.3f} {med(1):>6.3f} "
            f"{med(2):>7.3f} {med(3):>6.3f} {horiz:>5.0%}"
        )


if __name__ == "__main__":
    main()
