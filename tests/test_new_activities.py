"""Falling down (skeleton) and open-vocabulary activities (X-CLIP voting)."""
from __future__ import annotations

import numpy as np

from src.action_recognition.fall_detector import LABEL, FallDetector
from src.action_recognition.open_vocab import OpenVocabRecognizer, _State


def _kp(angle_deg: float) -> np.ndarray:
    """17 COCO keypoints whose torso (shoulders -> hips) leans angle_deg."""
    kp = np.zeros((17, 3))
    a = np.radians(angle_deg)
    sx, sy = 100.0, 100.0
    hx, hy = sx + 100 * np.sin(a), sy + 100 * np.cos(a)
    kp[5] = kp[6] = (sx, sy, 0.9)
    kp[11] = kp[12] = (hx, hy, 0.9)
    return kp


STANDING = ((100, 0, 160, 200), _kp(5))      # aspect 0.3, torso upright
LYING = ((0, 150, 220, 200), _kp(80))        # aspect 4.4, torso horizontal
SIT_EDGE = ((60, 100, 180, 220), _kp(30))     # aspect 1.0: sitting on a bed edge


def _run(seq, fps=10, pid=1, det=None):
    det = det or FallDetector()
    out = []
    for i, (box, kp) in enumerate(seq):
        r = det.update(pid, box, kp, i / fps)
        out.append(r.label if r else None)
    return out


def test_fall_is_detected_after_confirmation():
    out = _run([STANDING] * 10 + [LYING] * 10)
    assert LABEL in out
    assert out[:14].count(LABEL) == 0          # not before 0.4 s of lying


def test_slow_lie_down_is_not_a_fall():
    # Upright, then 2 s in between (sitting on the edge), then lying.
    out = _run([STANDING] * 10 + [SIT_EDGE] * 20 + [LYING] * 20)
    assert LABEL not in out


def test_lying_without_being_seen_upright_is_not_a_fall():
    assert LABEL not in _run([LYING] * 30)


def test_box_shape_alone_is_not_enough():
    wide_no_torso = ((0, 150, 220, 200), None)
    assert LABEL not in _run([STANDING] * 10 + [wide_no_torso] * 20)


def test_fall_holds_then_clears_after_getting_up():
    out = _run([STANDING] * 10 + [LYING] * 30 + [STANDING] * 80)
    assert out[45] == LABEL                     # still down / within hold
    assert out[-1] is None                      # up again for a while


def test_new_track_id_mid_fall_keeps_history():
    det = FallDetector()
    for i in range(10):
        det.update(1, STANDING[0], STANDING[1], i / 10)
    # The tracker renumbers the falling body: same place, new ID.
    out = [det.update(7, LYING[0], LYING[1], 1.0 + i / 10) for i in range(10)]
    assert any(r is not None and r.label == LABEL for r in out)


def _recognizer(threshold=0.35, min_hits=2) -> OpenVocabRecognizer:
    r = OpenVocabRecognizer.__new__(OpenVocabRecognizer)
    r.activities = {"taking tablets": ["taking a pill"], "making tea": ["making tea"]}
    r.enabled, r.threshold, r.min_hits, r.hold_s = True, threshold, min_hits, 4.0
    r._ready = True
    r.thresholds = {a: threshold for a in r.activities}
    r.strong = 0.5
    import threading
    r._states, r._lock = {0: _State()}, threading.Lock()
    return r


def test_open_vocab_needs_repeated_looks():
    r = _recognizer()
    r._vote(0, {"taking tablets": 0.45, "making tea": 0.01})
    assert r.current(1, single_person=True) is None        # one ordinary look is not enough
    r._vote(0, {"taking tablets": 0.40, "making tea": 0.02})
    p = r.current(1, single_person=True)
    assert p.label == "taking tablets" and 0.5 < p.confidence <= 0.95


def test_open_vocab_ignores_weak_looks():
    r = _recognizer()
    for _ in range(3):
        r._vote(0, {"taking tablets": 0.2, "making tea": 0.1})   # e.g. someone drinking
    assert r.current(1, single_person=True) is None


def test_open_vocab_off_without_gpu():
    from config import settings
    if settings.device.startswith("cuda"):
        return
    assert OpenVocabRecognizer().enabled is False


def test_open_vocab_one_strong_look_confirms():
    r = _recognizer()
    r._vote(0, {"taking tablets": 0.05, "making tea": 0.78})
    assert r.current(1, single_person=True).label == "making tea"
