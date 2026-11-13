"""Does squashing portrait video into a square break classification?

The pipeline resized every frame to 224x224 with a plain cv2.resize, which
does NOT preserve aspect ratio. For a 478x850 portrait phone video that
stretches the subject ~1.8x horizontally — a distortion the model has never
seen in training.

This compares plain squash against aspect-preserving letterbox on the user's
own uploaded clips.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.action_recognition.target_vocabulary import resolve_indices  # noqa: E402

MEAN = (0.43216, 0.394666, 0.37645)
STD = (0.22803, 0.22145, 0.216989)


def squash(img, size=224):
    """What the pipeline did: distorts aspect ratio."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def letterbox(img, size=224):
    """Aspect-preserving: scale to fit, pad the remainder."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((size, size, 3), dtype=img.dtype)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out


def classify(model, allowed_t, allowed, idx2cat, id2label, frames):
    rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames])
    x = torch.from_numpy(rgb).float().div_(255.0).permute(3, 0, 1, 2).unsqueeze(0)
    x = (x - torch.tensor(MEAN).view(1, 3, 1, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1, 1)
    with torch.no_grad():
        probs = torch.softmax(model(x)[0], dim=-1)
        ap = probs[allowed_t]
        cov = float(ap.sum())
        r = ap / ap.sum()
        b = int(torch.argmax(r))
        top = torch.topk(probs, 1)
    return (idx2cat[allowed[b]], float(r[b]), cov,
            id2label[int(top.indices[0])], float(top.values[0]))


def main() -> None:
    from torchvision.models.video import S3D_Weights, s3d
    w = S3D_Weights.KINETICS400_V1
    model = s3d(weights=w).eval()
    id2label = {i: n for i, n in enumerate(w.meta["categories"])}
    allowed, idx2cat = resolve_indices(id2label)
    allowed_t = torch.tensor(allowed, dtype=torch.long)

    clips = sorted(Path("data/samples").glob("WhatsApp*"))
    seen = set()
    print(f"  {'clip':<26}{'SQUASH (current)':<34}{'LETTERBOX (fixed)':<34}")
    print("  " + "-" * 96)

    for path in clips:
        # skip the duplicate _1/_2/_3 copies
        stem = path.stem.rstrip("_0123456789")
        if stem in seen:
            continue
        seen.add(stem)

        cap = cv2.VideoCapture(str(path))
        sq, lb = [], []
        while len(sq) < 16:
            ok, f = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, f = cap.read()
                if not ok:
                    break
            sq.append(squash(f))
            lb.append(letterbox(f))
        cap.release()
        if len(sq) < 16:
            continue

        s_lbl, s_c, s_cov, s_raw, s_rawc = classify(
            model, allowed_t, allowed, idx2cat, id2label, sq[:16])
        l_lbl, l_c, l_cov, l_raw, l_rawc = classify(
            model, allowed_t, allowed, idx2cat, id2label, lb[:16])

        a = f"{s_lbl} {s_c:.2f} cov{s_cov:.2f}"
        b = f"{l_lbl} {l_c:.2f} cov{l_cov:.2f}"
        mark = "  <-- BETTER" if l_cov > s_cov * 1.25 else ""
        print(f"  {path.stem[-24:]:<26}{a:<34}{b:<34}{mark}")
        print(f"  {'':<26}{'raw: ' + s_raw[:24]:<34}{'raw: ' + l_raw[:24]:<34}")

    print("\n  cov = how much belief lands in our 11 target categories.")


if __name__ == "__main__":
    main()
