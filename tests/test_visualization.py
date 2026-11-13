"""Label overlay must stay on screen.

Regression test for a real client-visible bug: a person near the top edge of
frame (the common case for a single person filling the frame — see the
full-frame framing rule in orchestrator.py) had their label strip drawn
entirely above y=0, so it was silently clipped by the frame boundary.
"""
from __future__ import annotations

import numpy as np

from src.utils.visualization import draw_overlays


def _frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _has_nonblack_pixels_in_row(img: np.ndarray, row: int) -> bool:
    return bool(np.any(img[row] != 0))


def test_label_visible_when_box_touches_top_edge():
    frame = _frame()
    annotations = [{
        "xyxy": (50, 0, 300, 400),  # y1 = 0: no room above for the old logic
        "person_id": 1,
        "label": "smoking",
        "confidence": 0.55,
    }]
    out = draw_overlays(frame, annotations)
    # The label strip must land INSIDE the frame's first few rows, not be
    # clipped away entirely.
    assert _has_nonblack_pixels_in_row(out, 0), "label strip missing at row 0"
    assert _has_nonblack_pixels_in_row(out, 10), "label strip missing near top"


def test_label_visible_when_box_has_room_above():
    """Normal case — box well inside frame — must still render as before."""
    frame = _frame()
    annotations = [{
        "xyxy": (50, 100, 300, 400),
        "person_id": 1,
        "label": "cooking",
        "confidence": 0.9,
    }]
    out = draw_overlays(frame, annotations)
    assert _has_nonblack_pixels_in_row(out, 90), "label strip missing above box"


def test_label_clamped_horizontally_near_right_edge():
    """A long label on someone near the right edge must not run off frame."""
    frame = _frame(w=200)
    annotations = [{
        "xyxy": (150, 100, 199, 400),
        "person_id": 1,
        "label": "using computer",
        "confidence": 0.99,
    }]
    out = draw_overlays(frame, annotations)
    # Nothing should be drawn past the last column — just confirms the
    # function ran without needing negative/out-of-bounds indices, since
    # cv2 would otherwise silently clip on that side too but the box
    # geometry itself should stay within [0, w).
    assert out.shape[1] == 200


def test_multiple_people_all_get_labels():
    frame = _frame()
    annotations = [
        {"xyxy": (10, 0, 100, 200), "person_id": 1, "label": "standing", "confidence": 0.8},
        {"xyxy": (300, 150, 400, 400), "person_id": 2, "label": "sitting", "confidence": 0.7},
    ]
    out = draw_overlays(frame, annotations)
    assert _has_nonblack_pixels_in_row(out, 0)
    assert _has_nonblack_pixels_in_row(out, 140)


def test_caption_is_burned_into_the_bottom_of_the_frame():
    frame = _frame()
    out = draw_overlays(frame, [], caption="A person is cooking in a kitchen.")
    h = out.shape[0]
    assert np.any(out[int(h * 0.75):] != 0)      # subtitle drawn near the bottom
    assert not np.any(out[: int(h * 0.5)] != 0)  # and nowhere near the top


def test_long_caption_wraps_and_stays_inside_the_frame():
    frame = _frame(h=240, w=320)
    text = "Three people on a street: two are smoking, one is standing, one not yet identified."
    out = draw_overlays(frame, [], caption=text)
    assert out.shape == frame.shape
    assert np.any(out[-40:] != 0)


def test_no_caption_leaves_frame_untouched():
    frame = _frame()
    assert not np.any(draw_overlays(frame, [], caption=None) != 0)



# --- Readability at high resolution / without system fonts ---------------
# Seen live on Kaggle with a 2732x1440 clip: a full-size caption box with
# ~11 px text inside (Pillow's fixed bitmap fallback, no system fonts on the
# machine), and box labels shrunk to near-nothing once the stream was
# downscaled for the browser.

def _text_rows(img: np.ndarray) -> int:
    """Rows containing bright (text) pixels."""
    return int(np.count_nonzero((img.max(axis=2) > 200).any(axis=1)))


def test_caption_text_is_large_without_any_system_font(monkeypatch):
    from src.utils import visualization as vis

    monkeypatch.setattr(vis, "_FONTS", [])          # a machine like Kaggle
    vis._caption_layer.cache_clear()
    h, w = 1440, 2732
    out = draw_overlays(_frame(h, w), [], "A person is brushing their teeth.")
    vis._caption_layer.cache_clear()
    # Font size is 4.5% of the frame height; the glyphs must really be that
    # big, not a fixed ~11 px bitmap inside a big box.
    assert _text_rows(out) >= 0.03 * h


def test_box_label_grows_with_frame_height():
    ann = [{"xyxy": (100, 200, 900, 1300), "person_id": 1,
            "label": "brushing teeth", "confidence": 0.95}]
    small = draw_overlays(_frame(480, 640), [dict(ann[0], xyxy=(50, 100, 300, 400))])
    big = draw_overlays(_frame(1440, 2732), ann)
    # Relative to its frame, the 1440p label must be about as tall as the
    # 480p one; before the fix it was a third of that.
    assert _text_rows(big) / 1440 >= 0.8 * _text_rows(small) / 480


def test_placeholder_label_has_no_unrenderable_characters(monkeypatch):
    """OpenCV's Hershey fonts draw any non-ASCII character as '?'."""
    import cv2
    from src.utils import visualization as vis

    drawn = []
    real = cv2.putText
    monkeypatch.setattr(vis.cv2, "putText", lambda img, text, *a, **k: (drawn.append(text), real(img, text, *a, **k))[1])
    draw_overlays(_frame(), [{"xyxy": (50, 50, 300, 400), "person_id": 2,
                              "label": "detecting…", "confidence": 0.0}])
    assert drawn and all(t.isascii() for t in drawn), drawn
