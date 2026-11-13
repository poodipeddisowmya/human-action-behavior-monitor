"""Why didn't the pose rules fire on these clips?

Replays a clip through the tracker + pose classifier and prints, per frame,
the geometric quantities each rule tests. Turns "the label is wrong" into
"this specific threshold was not met, by this much".
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.action_recognition import action_classifier as AC  # noqa: E402
from src.tracking import PersonTracker  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()

    path = Path("data/samples") / args.clip
    tracker = PersonTracker()
    pose = AC.SkeletonHeuristicClassifier()

    cap = cv2.VideoCapture(str(path))
    labels = Counter()
    rows = []
    n = 0
    while n < args.frames:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        tracks = tracker.update(f)
        if not tracks:
            continue
        t = tracks[0]
        pred = pose.update(t.person_id, t.keypoints, t.xyxy, None)
        labels[pred.label if pred else "(none)"] += 1

        kp = t.keypoints
        if kp is None or n % 8 != 0:
            continue

        plausible = AC._skeleton_is_plausible(kp)
        bh = max(1.0, t.xyxy[3] - t.xyxy[1])
        hx, hy, hv = AC._mid(kp, AC.L_HIP, AC.R_HIP)
        sx, sy, sv = AC._mid(kp, AC.L_SHOULDER, AC.R_SHOULDER)
        ax, ay, av = AC._mid(kp, AC.L_ANKLE, AC.R_ANKLE)
        kx, ky, kv = AC._mid(kp, AC.L_KNEE, AC.R_KNEE)
        torso = abs(hy - sy) / bh
        legs = abs(ay - hy) / bh if av >= AC._MIN_VIS else -1
        nose_x, nose_y, nose_v = kp[AC.NOSE]
        wl, wr = kp[AC.L_WRIST], kp[AC.R_WRIST]
        d_l = math.hypot(wl[0] - nose_x, wl[1] - nose_y) / bh if nose_v > 0 else 9
        d_r = math.hypot(wr[0] - nose_x, wr[1] - nose_y) / bh if nose_v > 0 else 9

        rows.append(
            f"  f{n:<4} plausible={str(plausible):<5} "
            f"torso={torso:.2f} legs={legs:>5.2f} "
            f"(SIT: legs<0.28 & torso>0.18) "
            f"wristnose L={d_l:.2f} R={d_r:.2f} "
            f"(DRINK needs <0.25) wvis={wl[2]:.2f}/{wr[2]:.2f}"
        )

    cap.release()
    print(f"\n{args.clip}  ({n} frames)")
    print("  pose labels:", dict(labels))
    print()
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
