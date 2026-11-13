"""Side-by-side BEFORE / AFTER videos from a GPU showcase recording.

    python scripts/build_comparison.py dist/client-demo

For every clip recorded under both builds (kaggle/gpu_showcase.py writes the
previous build's run to before/), composes one video: previous build on the
left, current build on the right, frame-locked, under a header band that
names each side and shows the measured figure for that clip. Writes
<clip>_compare.mp4 and comparison.json next to showcase.json.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.visualization import _font  # noqa: E402

SIDE_W = 640                  # each half; the composite is 1280 wide
BAND_H = 76
GAP = 4
INK = (255, 255, 255)
MUTED = (170, 178, 196)
BEFORE_TAG = (148, 64, 64)    # muted red
AFTER_TAG = (34, 139, 84)     # green
BAND_BG = (17, 24, 39)


def metric(s: dict) -> tuple[str, str]:
    """(headline figure, caption) for one side."""
    exp = s.get("expected")
    if exp:
        share = (s.get("labels_shown") or {}).get(exp, 0.0)
        first = (s.get("first_seen_s") or {}).get(exp)
        when = f"first shown at {first:.1f} s" if first is not None else "never shown"
        return f"{share:.0%}", f"correct label ({exp}) on screen · {when}"
    return f"{s.get('tracked_mean', 0):.1f}", "people tracked per frame (average)"


def band(before: dict, after: dict, left: dict, right: dict) -> np.ndarray:
    w = SIDE_W * 2 + GAP
    img = Image.new("RGB", (w, BAND_H), BAND_BG)
    d = ImageDraw.Draw(img)
    tag_f, big_f, cap_f = _font(15), _font(26), _font(14)
    for x0, (tag, colour, sub), s in (
            (0, (left["tag"], BEFORE_TAG, left["sub"]), before),
            (SIDE_W + GAP, (right["tag"], AFTER_TAG, right["sub"]), after)):
        tw = d.textlength(tag, font=tag_f)
        d.rounded_rectangle((x0 + 14, 12, x0 + 14 + tw + 16, 34), radius=11, fill=colour)
        d.text((x0 + 22, 14), tag, font=tag_f, fill=INK)
        d.text((x0 + 14 + tw + 26, 14), sub, font=tag_f, fill=MUTED)
        fig, cap = metric(s)
        d.text((x0 + 14, 40), fig, font=big_f, fill=INK)
        d.text((x0 + 14 + d.textlength(fig, font=big_f) + 10, 50), cap, font=cap_f, fill=MUTED)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def fit(frame: np.ndarray, h: int) -> np.ndarray:
    fh, fw = frame.shape[:2]
    scale = min(SIDE_W / fw, h / fh)
    small = cv2.resize(frame, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)
    out = np.zeros((h, SIDE_W, 3), np.uint8)
    y, x = (h - small.shape[0]) // 2, (SIDE_W - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    return out


def compose(left: Path, right: Path, top: np.ndarray, out: Path) -> bool:
    a, b = cv2.VideoCapture(str(left)), cv2.VideoCapture(str(right))
    fps = b.get(cv2.CAP_PROP_FPS) or 25.0
    h = int(SIDE_W * b.get(cv2.CAP_PROP_FRAME_HEIGHT) / max(1, b.get(cv2.CAP_PROP_FRAME_WIDTH)))
    h -= h % 2
    raw = out.with_suffix(".raw.mp4")
    w = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (SIDE_W * 2 + GAP, BAND_H + h))
    n = 0
    while True:
        ok1, f1 = a.read()
        ok2, f2 = b.read()
        if not (ok1 and ok2):
            break
        row = np.hstack([fit(f1, h), np.full((h, GAP, 3), 40, np.uint8), fit(f2, h)])
        w.write(np.vstack([top, row]))
        n += 1
    w.release()
    if not n:
        raw.unlink(missing_ok=True)
        return False
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg and subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(raw),
                                  "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                                  "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                  str(out)]).returncode == 0:
        raw.unlink(missing_ok=True)
    else:
        raw.replace(out)
    return True


def build(folder: Path) -> list[dict]:
    data = json.loads((folder / "showcase.json").read_text(encoding="utf-8"))
    after = {Path(s["video"]).stem: s for s in data.get("clips", [])}
    # Side labels: set_baseline.py writes WITHOUT GPU / WITH GPU; a plain
    # GPU showcase compares the previous build with the current one.
    left = data.get("compare_left") or {"tag": "BEFORE", "sub": "previous build"}
    right = data.get("compare_right") or {"tag": "AFTER", "sub": "GPU build"}
    pairs = []
    wanted = data.get("compare")            # None: every clip recorded both ways
    for b in data.get("before", []):
        stem = Path(b["video"]).stem
        if wanted is not None and stem not in wanted:
            continue
        a = after.get(stem)
        if not a or not b.get("annotated_video") or not a.get("annotated_video"):
            continue
        out = folder / f"{stem}_compare.mp4"
        if compose(folder / "before" / b["annotated_video"], folder / a["annotated_video"],
                   band(b, a, left, right), out):
            bf, af = metric(b)[0], metric(a)[0]
            pairs.append({"clip": stem, "video": out.name, "expected": a.get("expected"),
                          "before": bf, "after": af,
                          "before_first_s": (b.get("first_seen_s") or {}).get(a.get("expected")),
                          "after_first_s": (a.get("first_seen_s") or {}).get(a.get("expected")),
                          "caption": metric(a)[1].split(" · ")[0]})
            print(f"  {stem:<28} before {bf:>5}  after {af:>5}")
    (folder / "comparison.json").write_text(json.dumps(pairs, indent=2))
    return pairs


if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else "dist/client-demo"))
