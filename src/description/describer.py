"""The one object the pipeline talks to for scene description."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .captions import CaptionComposer
from .scene import SceneClassifier

NARRATE_MIN_HEIGHT = 360        # px; see SceneDescriber.update

# Labels that describe a body position rather than an activity.
_POSTURES = {"standing", "sitting", "walking", "running", "lying", "kneeling",
             "bending", "crawling", "detecting…"}


class SceneDescriber:
    def __init__(self, scene: bool = True, narrator=None):
        self._scene = SceneClassifier() if scene else None
        self._captions = CaptionComposer()
        # Optional detailed narration (narrator.py, GPU). When it has a fresh
        # description it is shown; otherwise the template caption is.
        self._narrator = narrator if narrator is not None and narrator.available else None

    def update(self, frame: np.ndarray, annotations: list[dict]) -> dict:
        """Call once per processed frame; returns {"text", "scene"}."""
        phrase = None
        if self._scene is not None and self._scene.available:
            self._scene.submit(frame)
            phrase = self._scene.phrase
        text = self._captions.update(annotations, phrase)
        # Detailed narration needs detail to look at. On a 320x240 clip the
        # model invented "using a blue shirt as a broom handle" (T4 run); below
        # this height the template caption, built from what was detected, is
        # the honest one.
        if self._narrator is not None and frame.shape[0] >= NARRATE_MIN_HEIGHT:
            acts = Counter(a.get("label") for a in annotations
                           if a.get("label") and a.get("label") not in _POSTURES)
            self._narrator.submit(frame, len(annotations),
                                  acts.most_common(1)[0][0] if acts else None, phrase)
            text = self._narrator.text or text
        return {"text": text, "scene": phrase}

    @property
    def current(self) -> dict:
        phrase = self._scene.phrase if self._scene is not None else None
        text = (self._narrator.text if self._narrator is not None else None) or self._captions.text
        return {"text": text, "scene": phrase}

    def reset(self) -> None:
        self._captions.reset()
        if self._scene is not None:
            self._scene.reset()
        if self._narrator is not None:
            self._narrator.reset()
