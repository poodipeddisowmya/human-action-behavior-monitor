"""Profile where per-frame time actually goes, with and without a tracked person.

The pipeline drops from ~13 FPS (no people) to ~4 FPS (one person). This
isolates which stage is responsible rather than guessing.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402


def main() -> None:
    print(f"torch threads = {torch.get_num_threads()}\n")

    from src.tracking import PersonTracker
    from src.action_recognition.action_classifier import SkeletonHeuristicClassifier
    from src.utils import draw_overlays

    clip = Path("data/samples/demo-exercising-squats.avi")
    cap = cv2.VideoCapture(str(clip))
    frames = []
    while len(frames) < 40:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    print(f"loaded {len(frames)} frames from {clip.name}\n")

    tracker = PersonTracker()
    pose = SkeletonHeuristicClassifier()

    t_track = t_pose = t_crop = t_draw = t_jpeg = 0.0
    n = 0
    for f in frames:
        t0 = time.perf_counter()
        tracks = tracker.update(f)
        t1 = time.perf_counter()

        anns = []
        h, w = f.shape[:2]
        for tr in tracks:
            x1, y1, x2, y2 = map(int, tr.xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            ta = time.perf_counter()
            crop = f[y1:y2, x1:x2]
            _ = cv2.resize(crop, (224, 224))
            tb = time.perf_counter()
            pose.update(tr.person_id, tr.keypoints, tr.xyxy, crop)
            tc = time.perf_counter()
            t_crop += tb - ta
            t_pose += tc - tb
            anns.append({"xyxy": tr.xyxy, "person_id": tr.person_id,
                         "label": "x", "confidence": 0.5})

        t2 = time.perf_counter()
        out = draw_overlays(f, anns)
        t3 = time.perf_counter()
        cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        t4 = time.perf_counter()

        t_track += t1 - t0
        t_draw += t3 - t2
        t_jpeg += t4 - t3
        n += 1

    print(f"  {'stage':<28}{'ms/frame':>10}")
    print("  " + "-" * 38)
    for name, tot in [("tracker (YOLO+ByteTrack)", t_track),
                      ("crop + resize", t_crop),
                      ("pose classifier", t_pose),
                      ("draw overlays", t_draw),
                      ("jpeg encode", t_jpeg)]:
        print(f"  {name:<28}{tot/n*1000:>7.1f} ms")
    total = (t_track + t_crop + t_pose + t_draw + t_jpeg) / n * 1000
    print(f"  {'TOTAL':<28}{total:>7.1f} ms   -> {1000/total:.1f} FPS ceiling")

    # --- now measure the same while a background s3d inference runs ---
    print("\n  With s3d inference running concurrently (thread contention test):")
    from torchvision.models.video import S3D_Weights, s3d
    model = s3d(weights=S3D_Weights.KINETICS400_V1).eval()
    x = torch.randn(1, 3, 16, 224, 224)

    import threading
    stop = threading.Event()

    def hammer():
        while not stop.is_set():
            with torch.no_grad():
                model(x)

    th = threading.Thread(target=hammer, daemon=True)
    th.start()
    time.sleep(1.0)

    t0 = time.perf_counter()
    for f in frames[:20]:
        tracker.update(f)
    contended = (time.perf_counter() - t0) / 20 * 1000
    stop.set()
    th.join(timeout=5)

    print(f"    tracker alone      {t_track/n*1000:>7.1f} ms")
    print(f"    tracker contended  {contended:>7.1f} ms"
          f"   ({contended/(t_track/n*1000):.1f}x slower)")


if __name__ == "__main__":
    main()
