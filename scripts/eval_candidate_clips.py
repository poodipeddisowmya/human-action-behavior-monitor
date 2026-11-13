"""Screen candidate reference clips before they're adopted as demo footage.

Checks the two things that actually matter for THIS pipeline, not a generic
"is this a good video" opinion:

  1. Does YOLOv8-pose detect a person at all? A clip can be a perfectly good
     stock video and still be useless here if it's a hands-only close-up with
     no head/torso for the detector to lock onto.
  2. Does the deep classifier — fed exactly the way the orchestrator feeds it
     (full frame for a single person, see _process_tracks) — land on the
     target category, and with how much coverage?

    python scripts/eval_candidate_clips.py --clip path/to.mp4 --expect smoking
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.action_recognition import ActionClassifier  # noqa: E402
from src.tracking import PersonTracker  # noqa: E402


def run_clip(path: Path, max_frames: int = 200) -> dict:
    tracker = PersonTracker()
    classifier = ActionClassifier()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frames_with_person = 0
    n = 0
    labels: dict[str, int] = {}
    max_conf: dict[str, float] = {}
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        ts = (n - 1) / fps
        h, w = frame.shape[:2]
        tracks = tracker.update(frame)
        if tracks:
            frames_with_person += 1
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.xyxy)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            crop = frame if len(tracks) == 1 else frame[y1:y2, x1:x2]
            pred = classifier.update(t.person_id, t.keypoints, t.xyxy, crop, ts)
            if pred is not None:
                labels[pred.label] = labels.get(pred.label, 0) + 1
                max_conf[pred.label] = max(max_conf.get(pred.label, 0.0), pred.confidence)
    cap.release()
    return {
        "frames": n,
        "frames_with_person": frames_with_person,
        "labels": labels,
        "max_conf": max_conf,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--expect", default=None, help="target category, for a pass/fail mark")
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()

    path = Path(args.clip)
    result = run_clip(path, args.frames)

    detect_rate = result["frames_with_person"] / result["frames"] if result["frames"] else 0
    print(f"\n  {path.name}")
    print(f"  person detected: {result['frames_with_person']}/{result['frames']} frames "
          f"({detect_rate:.0%})")
    if not result["labels"]:
        print("  labels: (none — no usable person/prediction)")
    else:
        for label, count in sorted(result["labels"].items(), key=lambda x: -x[1]):
            print(f"  label={label:<16} seen {count:>3}x   best_conf={result['max_conf'][label]:.2f}")

    if args.expect:
        top = max(result["labels"].items(), key=lambda x: x[1])[0] if result["labels"] else None
        verdict = "PASS" if top == args.expect else "FAIL"
        print(f"  expect={args.expect} got={top}  [{verdict}]")


if __name__ == "__main__":
    main()
