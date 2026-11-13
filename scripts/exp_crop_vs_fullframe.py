"""Does classifying the FULL FRAME beat classifying the person crop?

Kinetics-400 models are trained on whole scenes — a kitchen, a gym, a desk —
not on tight person crops. Feeding a crop may strip exactly the context the
model relies on ("is there a stove in shot?").

This measures both, per clip, and reports which wins.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.action_recognition.target_vocabulary import resolve_indices  # noqa: E402

MEAN = (0.43216, 0.394666, 0.37645)
STD = (0.22803, 0.22145, 0.216989)

EXPECTED = {
    "demo-cooking.mp4": "cooking",
    "demo-eating.avi": "eating",
    "demo-drinking.mp4": "drinking",
    "demo-writing.avi": "writing",
    "demo-using-computer.avi": "using computer",
    "demo-cleaning-mopping.avi": "cleaning",
    "demo-cycling.avi": "cycling",
    "demo-exercising-squats.avi": "exercising",
}


def classify(model, allowed_t, allowed, idx2cat, frames):
    """frames: list of 16 BGR uint8 arrays already at 224x224."""
    rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames])
    x = torch.from_numpy(rgb).float().div_(255.0).permute(3, 0, 1, 2).unsqueeze(0)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        probs = torch.softmax(model(x)[0], dim=-1)
        allowed_probs = probs[allowed_t]
        coverage = float(allowed_probs.sum())
        restricted = allowed_probs / allowed_probs.sum()
        best = int(torch.argmax(restricted))
    return idx2cat[allowed[best]], float(restricted[best]), coverage


def main() -> None:
    from torchvision.models.video import S3D_Weights, s3d
    from ultralytics import YOLO

    w = S3D_Weights.KINETICS400_V1
    model = s3d(weights=w).eval()
    id2label = {i: n for i, n in enumerate(w.meta["categories"])}
    allowed, idx2cat = resolve_indices(id2label)
    allowed_t = torch.tensor(allowed, dtype=torch.long)
    det = YOLO("yolov8n-pose.pt")

    print(f"  {'clip':<28}{'expected':<16}{'CROP':<26}{'FULL FRAME':<26}{'winner'}")
    print("  " + "-" * 108)

    crop_wins = full_wins = 0
    for clip, expected in EXPECTED.items():
        path = Path("data/samples") / clip
        if not path.exists():
            continue
        cap = cv2.VideoCapture(str(path))
        crops, fulls = [], []
        while len(fulls) < 16:
            ok, f = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, f = cap.read()
                if not ok:
                    break
            fulls.append(cv2.resize(f, (224, 224)))
            r = det.predict(f, imgsz=416, conf=0.25, verbose=False, device="cpu")[0]
            if r.boxes is not None and len(r.boxes):
                x1, y1, x2, y2 = map(int, r.boxes[0].xyxy[0].tolist())
                h, w_ = f.shape[:2]
                px, py = int((x2 - x1) * .25), int((y2 - y1) * .25)
                c = f[max(0, y1-py):min(h, y2+py), max(0, x1-px):min(w_, x2+px)]
                if c.size:
                    crops.append(cv2.resize(c, (224, 224)))
        cap.release()
        if len(crops) < 16 or len(fulls) < 16:
            print(f"  {clip:<28}{expected:<16}(not enough frames)")
            continue

        cl, cc, ccov = classify(model, allowed_t, allowed, idx2cat, crops[:16])
        fl, fc, fcov = classify(model, allowed_t, allowed, idx2cat, fulls[:16])

        c_ok, f_ok = cl == expected, fl == expected
        winner = "FULL" if (f_ok and not c_ok) else "CROP" if (c_ok and not f_ok) else "tie"
        if winner == "FULL":
            full_wins += 1
        elif winner == "CROP":
            crop_wins += 1

        cs = f"{cl} {cc:.2f} cov{ccov:.2f}{' OK' if c_ok else ''}"
        fs = f"{fl} {fc:.2f} cov{fcov:.2f}{' OK' if f_ok else ''}"
        print(f"  {clip:<28}{expected:<16}{cs:<26}{fs:<26}{winner}")

    print(f"\n  full-frame wins: {full_wins}   crop wins: {crop_wins}")


if __name__ == "__main__":
    main()
