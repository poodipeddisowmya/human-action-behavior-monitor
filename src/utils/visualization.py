"""Draw boxes, IDs, action labels and the scene caption on video frames."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

_COLORS = [
    (46, 204, 113), (52, 152, 219), (155, 89, 182), (241, 196, 15),
    (231, 76, 60), (26, 188, 156), (230, 126, 34), (149, 165, 166),
]


def _color(pid: int) -> tuple[int, int, int]:
    return _COLORS[pid % len(_COLORS)]


def _label_scale(frame_h: int) -> tuple[float, int]:
    """(font scale, stroke) for box labels.

    The video is downscaled for the browser (STREAM_MAX_WIDTH), so a fixed
    pixel size that reads well at 720p shrank to near-nothing on a 1440p
    clip. Unchanged up to 720p; grows with the frame above that.
    """
    scale = 0.6 * max(1.0, frame_h / 720)
    return scale, max(2, round(scale * 3.3))


def draw_overlays(frame: np.ndarray, annotations: list[dict],
                  caption: str | None = None) -> np.ndarray:
    """annotations: list of {xyxy, person_id, label, confidence}.

    `caption`, when given, is burned into the bottom of the frame as a
    subtitle, so it is part of the video itself (the MJPEG stream, a screen
    recording, a re-broadcast) rather than only of the web page around it.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    scale, stroke = _label_scale(h)
    pad = max(8, int(8 * scale / 0.6))
    for a in annotations:
        x1, y1, x2, y2 = map(int, a["xyxy"])
        c = _color(a["person_id"])
        cv2.rectangle(out, (x1, y1), (x2, y2), c, stroke)

        # Hershey fonts are ASCII-only: "detecting…" was drawn as
        # "detecting???" on every new person until the model had looked.
        label = str(a.get("label", "-")).replace("…", "...")
        # An enrolled person's name (face recognition) instead of the track
        # number; Hershey cannot draw non-ASCII, so such letters become "?".
        who = a.get("name")
        who = who.encode("ascii", "replace").decode() if who else f'P{a["person_id"]:03d}'
        text = f'{who}  {label}  {a.get("confidence",0):.0%}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, stroke)
        label_h = th + pad

        # Label is normally drawn in a strip just ABOVE the box's top edge.
        # For anyone standing close enough to fill the frame — exactly the
        # single-person full-frame case this system prefers, see the framing
        # note in orchestrator.py — that top edge sits at or near y=0, and
        # the whole strip was drawn at negative y: still "drawn", just
        # clipped by the frame boundary, so it read as a cut-off label.
        # When there's no room above, draw the strip just inside the box's
        # top edge instead.
        baseline = pad // 2
        if y1 - label_h >= 0:
            box_top, box_bottom, text_y = y1 - label_h, y1, y1 - baseline
        else:
            box_top, box_bottom, text_y = y1, y1 + label_h, y1 + label_h - baseline

        # Same idea horizontally: clamp so a long label on someone near the
        # right edge doesn't run off frame either.
        box_right = min(w, x1 + tw + pad)
        box_left = max(0, box_right - (tw + pad))

        cv2.rectangle(out, (box_left, box_top), (box_right, box_bottom), c, -1)
        cv2.putText(out, text, (box_left + pad // 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), stroke)
    if caption:
        _blend_caption(out, caption)
    return out


# --------------------------------------------------------------------------
# Subtitle
# --------------------------------------------------------------------------
# OpenCV's built-in Hershey fonts are blocky and have no "…"; a real
# TrueType face reads like broadcast subtitles. Looked up in order.
_FONTS = [
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
]


def _font(size: int):
    """A scalable font at `size` px, on any machine.

    The last resort used to be Pillow's load_default(): a fixed ~11 px
    bitmap that ignores `size`. On Kaggle (none of the fonts above) that
    drew a full-size caption box with microscopic text inside it.
    matplotlib — always installed, ultralytics needs it — ships DejaVu Sans,
    so a real TrueType face is available everywhere.
    """
    from PIL import ImageFont

    candidates = list(_FONTS)
    try:
        import matplotlib
        candidates.append(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
    except Exception:
        pass
    for f in candidates:
        if f.exists():
            try:
                return ImageFont.truetype(str(f), size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)      # scalable on Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


@lru_cache(maxsize=32)
def _caption_layer(text: str, frame_w: int, frame_h: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Render the subtitle once per (text, frame size): BGR pixels + alpha.

    Cached because the caption changes every few seconds at most while
    frames arrive ~25 times a second; per frame only a numpy blend runs.
    """
    from PIL import Image, ImageDraw

    size = max(14, int(frame_h * 0.045))
    font = _font(size)

    # Wrap to at most 90% of the frame width.
    max_w = int(frame_w * 0.9) - 2 * size
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and probe.textlength(trial, font=font) > max_w:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)

    pad_x, pad_y, gap = size, int(size * 0.45), int(size * 0.25)
    line_h = size + gap
    text_w = max(int(probe.textlength(l, font=font)) for l in lines)
    w = min(frame_w, text_w + 2 * pad_x)
    h = len(lines) * line_h - gap + 2 * pad_y + int(size * 0.2)

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=int(size * 0.35), fill=(0, 0, 0, 175))
    y = pad_y
    for l in lines:
        lw = d.textlength(l, font=font)
        d.text(((w - lw) / 2, y), l, font=font, fill=(255, 255, 255, 255))
        y += line_h
    rgba = np.asarray(img)
    bgr = rgba[:, :, 2::-1].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    x0 = (frame_w - w) // 2
    y0 = frame_h - h - int(frame_h * 0.05)
    return bgr, alpha, x0, max(0, y0)


def _blend_caption(out: np.ndarray, text: str) -> None:
    h, w = out.shape[:2]
    bgr, alpha, x0, y0 = _caption_layer(text, w, h)
    ch, cw = bgr.shape[:2]
    ch, cw = min(ch, h - y0), min(cw, w - x0)
    region = out[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
    a = alpha[:ch, :cw]
    out[y0:y0 + ch, x0:x0 + cw] = (bgr[:ch, :cw] * a + region * (1 - a)).astype(np.uint8)
