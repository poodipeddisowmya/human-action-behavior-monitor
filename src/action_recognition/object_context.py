"""Object detection + person-object interaction — a third evidence source.

Why this exists
----------------
The pose classifier (action_classifier.py) answers "what is the body doing":
standing, sitting, bending — pure geometry, no idea what's in the person's
hands. The deep scene classifier (videomae_classifier.py) answers "what does
this whole clip look like" by matching the full frame against Kinetics-400,
which is powerful but holistic — it can miss a specific, unambiguous cue that
a much simpler check would catch instantly. A toothbrush at someone's mouth
IS brushing teeth; a human doesn't need to weigh 400 possibilities to know
that. This module adds exactly that kind of cheap, specific, object-grounded
evidence, feeding it into the SAME vote/consensus machinery the deep
classifier already uses (see the integration point in videomae_classifier.py)
rather than building a second, competing decision path.

What it can and cannot detect
------------------------------
Runs a general YOLOv8 object detector over COCO's 80 classes — a SEPARATE
model from the pose detector (PersonTracker), which only knows the "person"
class. COCO covers a genuinely useful subset of our target activities:
laptop/keyboard/mouse, cell phone, bottle/wine glass/cup, fork/knife/spoon/
bowl plus common foods, oven/microwave/toaster/sink/refrigerator, book,
toothbrush.

COCO does NOT include several tools real activities involve — a mop, a
spatula, a cigarette have no COCO class at all. Cleaning and smoking
therefore get no boost from this module and continue to rely on the scene
classifier alone, exactly as before this was added. That is a real, honest
limit of a pretrained general-purpose detector, not a bug to chase: naming it
here is cheaper than someone rediscovering it by watching mopping footage get
no object evidence and wondering why.

Interaction, not presence
--------------------------
A laptop sitting untouched in the corner of a shot is not evidence anyone is
"using a computer". Every detected object is scored for how physically tied
to the tracked person it looks — wrist proximity when keypoints are visible,
falling back to bbox containment when they are not (rear view, occlusion,
low light: the exact cases the pose signal is weakest for, and exactly where
this module is meant to help). An object with no plausible interaction
contributes nothing rather than a false cue.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import settings

from .action_classifier import ActionPrediction, L_WRIST, R_WRIST, _MIN_VIS_HAND

# --- Object -> category mapping -----------------------------------------
#
# `base_weight` is how unambiguous the object is as evidence for that
# category, GIVEN clear physical interaction — not the same as detector
# confidence. A toothbrush at your mouth is almost always brushing teeth
# (high weight); a bowl in your hands could be many things (low weight).
# These were set by judgment, not measurement (unlike the pose thresholds
# elsewhere in this codebase, which were fit to real footage) — they are the
# first thing to revisit if a category built on a weak object turns out to
# misfire in practice. scripts/eval_offline.py and eval_candidate_clips.py
# are the tools to re-measure them with.
@dataclass(frozen=True)
class ObjectHint:
    category: str
    base_weight: float


_OBJECT_HINTS: dict[str, ObjectHint] = {
    "laptop":       ObjectHint("using computer", 0.55),
    "keyboard":     ObjectHint("using computer", 0.45),
    "mouse":        ObjectHint("using computer", 0.35),
    "cell phone":   ObjectHint("using phone", 0.55),
    "book":         ObjectHint("reading", 0.45),
    "toothbrush":   ObjectHint("brushing teeth", 0.70),
    "wine glass":   ObjectHint("drinking", 0.55),
    "cup":          ObjectHint("drinking", 0.40),
    "bottle":       ObjectHint("drinking", 0.35),
    "fork":         ObjectHint("eating", 0.35),
    "spoon":        ObjectHint("eating", 0.35),
    "bowl":         ObjectHint("eating", 0.30),
    "knife":        ObjectHint("eating", 0.25),
    "sandwich":     ObjectHint("eating", 0.45),
    "pizza":        ObjectHint("eating", 0.45),
    "cake":         ObjectHint("eating", 0.40),
    "donut":        ObjectHint("eating", 0.45),
    "hot dog":      ObjectHint("eating", 0.45),
    "banana":       ObjectHint("eating", 0.40),
    "apple":        ObjectHint("eating", 0.40),
    "orange":       ObjectHint("eating", 0.35),
    "broccoli":     ObjectHint("eating", 0.35),
    "carrot":       ObjectHint("eating", 0.35),
    "oven":         ObjectHint("cooking", 0.40),
    "toaster":      ObjectHint("cooking", 0.35),
    "microwave":    ObjectHint("cooking", 0.30),
    "refrigerator": ObjectHint("cooking", 0.20),
    "sink":         ObjectHint("cleaning", 0.25),
}

# YOLO's own confidence floor for a detection to be considered at all — below
# this it's discarded before interaction scoring even runs.
_DET_CONF = 0.35

# Input size for the object detector. Measured on this machine across 640/
# 416/320/256 (scripts show the numbers in the object-context regression
# note below); 320 was the fastest and none showed a real accuracy cliff for
# objects that are typically a decent fraction of the frame (a laptop, a
# phone, a toothbrush near a face) rather than tiny/distant. This runs as a
# SECOND full detector alongside the pose tracker and the deep scene
# classifier — first shipped at full resolution, it measurably slowed the
# worker's per-cycle cost enough to regress three previously-passing basic-
# posture clips (demo-basic-sit/walk/run) in scripts/eval_offline.py, because
# a slower cycle means "detecting…" can't clear as fast as a churning track
# ID reassigns. Object presence is coarse, slow-changing evidence by nature
# (a laptop doesn't vanish from frame between one look and the next) —
# it does not need the resolution a fast-moving pose estimate does.
_DET_IMGSZ = 320

# How close an object's center must be to a visible wrist, in body-heights,
# to count as "in hand". 0.35 is generous enough to survive motion blur and
# imprecise boxes without being so loose it catches an object on a nearby
# table — a mid-range guess, not a measured constant; tighten it first if
# false interactions show up on real footage.
_HAND_RADIUS = 0.35

# Interaction tiers. Full confidence needs a visible hand near the object;
# the object-inside-person-bbox fallback is deliberately weaker (0.5, not
# 1.0) because it is real but non-specific evidence — the object could be
# resting against the person rather than held.
_INTERACT_HAND = 1.0
_INTERACT_COLOCATED = 0.5

# Reported vote confidence is capped here, same spirit as
# _MAX_REPORTED_CONF in videomae_classifier.py: a single object detection
# should never claim near-certainty. It still has to clear the shared
# _tally()'s evidence bar like any other vote — see the integration point.
_MAX_VOTE_CONF = 0.85


@dataclass
class _Detection:
    name: str
    xyxy: tuple[float, float, float, float]
    conf: float


def resolve_weights(name: str) -> str:
    """Same resolution as person_tracker.resolve_weights, duplicated locally
    to avoid this module depending on the tracking package for one helper."""
    if Path(name).is_absolute() and Path(name).exists():
        return name
    for candidate in (settings.models_dir / name, settings.root / name):
        if candidate.exists():
            return str(candidate)
    return name


class ObjectContextDetector:
    """Wraps a general-object YOLOv8 model and turns detections tied to a
    tracked person into ActionPrediction votes for the same category space
    the rest of the system uses.

    Deliberately mirrors VideoMAEClassifier's own graceful-degradation
    pattern: if the weights are missing, `self._model` stays None and
    `infer()` always returns None — callers do not need to know or care.
    """

    def __init__(self, weights: str = "yolov8n-objects.pt"):
        self._model = None
        try:
            from ultralytics import YOLO
            self._model = YOLO(resolve_weights(weights))
            names = self._model.model.names  # {index: coco_name}
            self._relevant_ids = {
                idx: name for idx, name in names.items() if name in _OBJECT_HINTS
            }
            print(f"[objects] loaded {weights} — tracking "
                  f"{len(self._relevant_ids)} of {len(names)} COCO classes: "
                  f"{', '.join(sorted(self._relevant_ids.values()))}", flush=True)
        except Exception as e:
            print(f"[objects] unavailable, running without object context. ({e})",
                  flush=True)
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    def _detect_relevant(self, frame_bgr: np.ndarray) -> list[_Detection]:
        results = self._model.predict(
            frame_bgr, conf=_DET_CONF, verbose=False, imgsz=_DET_IMGSZ,
            classes=list(self._relevant_ids), device=settings.device,
        )
        out: list[_Detection] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None:
            return out
        names = self._model.model.names
        for xyxy, conf, cls in zip(
            boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
        ):
            name = names[int(cls)]
            out.append(_Detection(name=name, xyxy=tuple(xyxy), conf=float(conf)))
        return out

    @staticmethod
    def _interaction(det: _Detection, person_bbox, keypoints, body_h: float) -> float:
        ox = (det.xyxy[0] + det.xyxy[2]) / 2
        oy = (det.xyxy[1] + det.xyxy[3]) / 2

        if keypoints is not None:
            for w_idx in (L_WRIST, R_WRIST):
                wx, wy, wv = keypoints[w_idx]
                if wv < _MIN_VIS_HAND:
                    continue
                dist = math.hypot(ox - wx, oy - wy) / max(body_h, 1.0)
                if dist <= _HAND_RADIUS:
                    return _INTERACT_HAND

        # No visible/near wrist — fall back to "is the object roughly on the
        # person at all", which survives rear views and occlusion (the exact
        # cases pose geometry is blind in) at the cost of being less specific.
        px1, py1, px2, py2 = person_bbox
        pad_x, pad_y = (px2 - px1) * 0.15, (py2 - py1) * 0.15
        if (px1 - pad_x <= ox <= px2 + pad_x) and (py1 - pad_y <= oy <= py2 + pad_y):
            return _INTERACT_COLOCATED
        return 0.0

    # ------------------------------------------------------------------
    def infer(
        self,
        frame_bgr: np.ndarray,
        person_bbox: tuple[float, float, float, float],
        keypoints: np.ndarray | None,
    ) -> ActionPrediction | None:
        if self._model is None or frame_bgr is None or frame_bgr.size == 0:
            return None

        body_h = max(1.0, person_bbox[3] - person_bbox[1])
        try:
            detections = self._detect_relevant(frame_bgr)
        except Exception as e:
            print(f"[objects] inference error: {e}", flush=True)
            return None

        # Score every relevant, interacting object; keep the strongest vote
        # per category rather than summing — two objects for the same
        # category (a fork AND a bowl) are corroborating, not independent,
        # evidence, and summing would let category with more possible object
        # classes (eating) win by object-count rather than by genuine
        # strength of evidence.
        best: dict[str, float] = {}
        best_name: dict[str, str] = {}
        for det in detections:
            hint = _OBJECT_HINTS[det.name]
            score = self._interaction(det, person_bbox, keypoints, body_h)
            if score <= 0.0:
                continue
            conf = min(_MAX_VOTE_CONF, hint.base_weight * score * det.conf)
            if conf > best.get(hint.category, 0.0):
                best[hint.category] = conf
                best_name[hint.category] = det.name

        if not best:
            return None
        category = max(best, key=best.get)
        pred = ActionPrediction(label=category, confidence=round(best[category], 3))
        # Which object it was ("cup", "laptop"): lets the caption say
        # "drinking from a cup" when this agrees with the confirmed activity.
        pred.detail = best_name[category]
        return pred
