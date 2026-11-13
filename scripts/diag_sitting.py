"""Why does the sitting rule not fire?

The rule needs ankles visible, which on a seated person is the single most
occluded joint (under a desk, behind a chair, cropped). This prints ankle
visibility and the knee-vs-hip geometry that does NOT depend on ankles, so a
replacement condition can be set from data.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--frames", type=int, default=100)
    args = ap.parse_args()

    print("\n  ankle_vis = fraction of frames with ankles above the visibility floor")
    print("  knee_hip  = (knee_y - hip_y) / body_height   [small = thighs level = seated]")
    print("  torso     = |hip_y - shoulder_y| / body_height\n")
    print(f"  {'clip':<32} {'n':>4} {'ankle_vis':>10} {'knee_hip':>9} {'torso':>7} {'legs':>7}")

    for name in args.clips:
        path = Path("data/samples") / name
        if not path.exists():
            print(f"  {name[:32]:<32} -- missing")
            continue
        tracker = PersonTracker()
        cap = cv2.VideoCapture(str(path))
        kh, tor, legs, avis = [], [], [], []
        n = 0
        while n < args.frames:
            ok, f = cap.read()
            if not ok:
                break
            n += 1
            for t in tracker.update(f):
                kp = t.keypoints
                if kp is None or not AC._skeleton_is_plausible(kp):
                    continue
                bh = max(1.0, t.xyxy[3] - t.xyxy[1])
                _, hy, hv = AC._mid(kp, AC.L_HIP, AC.R_HIP)
                _, sy, sv = AC._mid(kp, AC.L_SHOULDER, AC.R_SHOULDER)
                _, ay, av = AC._mid(kp, AC.L_ANKLE, AC.R_ANKLE)
                _, ky, kv = AC._mid(kp, AC.L_KNEE, AC.R_KNEE)
                if hv < AC._MIN_VIS or sv < AC._MIN_VIS or kv < AC._MIN_VIS:
                    continue
                avis.append(1.0 if av >= AC._MIN_VIS else 0.0)
                kh.append((ky - hy) / bh)
                tor.append(abs(hy - sy) / bh)
                if av >= AC._MIN_VIS:
                    legs.append(abs(ay - hy) / bh)
        cap.release()
        if not kh:
            print(f"  {name[:32]:<32} -- no usable skeletons")
            continue
        med = statistics.median
        print(
            f"  {name[:32]:<32} {len(kh):>4} {med(avis):>10.2f} {med(kh):>9.3f} "
            f"{med(tor):>7.3f} {(med(legs) if legs else float('nan')):>7.3f}"
        )
    print()


if __name__ == "__main__":
    main()
