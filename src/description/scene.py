"""Name the PLACE in view — kitchen, office, park — from text prompts.

Zero-shot CLIP (openai/clip-vit-base-patch32): the image is compared against
a short sentence for each place, so the list below is the whole vocabulary
and can be extended by adding a line. It runs on its own thread every few
seconds over the newest frame; the video and the action labels never wait
for it. Measured on this machine: ~0.1 s per look.

Precision over coverage. A wrong place in the subtitle is worse than none,
so a place is only named when it has won most recent looks with a clear
margin; otherwise the caption falls back to "indoors"/"outdoors" when that
much is certain, and to saying nothing when it is not. Measured on the 32
sample clips (3 frames each): kitchens, bathroom, gym, classroom, offices,
forest, courts and dining tables come out at 0.55-1.00; the confident errors
seen (a doorway pull-up read as "bathroom", 0.44-0.65) sit around the bar,
which the multi-look vote is there to absorb.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque

import numpy as np

from src.utils import get_logger

log = get_logger(__name__)

_CHECKPOINT = "openai/clip-vit-base-patch32"

# place -> (caption phrase, indoor?, prompts). indoor None = could be either.
_PLACES: dict[str, tuple[str, bool | None, list[str]]] = {
    "kitchen": ("in a kitchen", True,
                ["a photo of a kitchen", "a photo of a kitchen counter with a stove"]),
    "office": ("in an office", True,
               ["a photo of an office", "a photo of an office with desks and chairs"]),
    "living room": ("in a living room", True, ["a photo of a living room"]),
    "bedroom": ("in a bedroom", True, ["a photo of a bedroom"]),
    "bathroom": ("in a bathroom", True, ["a photo of a bathroom"]),
    "dining": ("at a dining table", True,
               ["a photo of a restaurant or cafe", "a photo of a dining table with food"]),
    "gym": ("in a gym", True, ["a photo of a gym", "a photo of a fitness studio"]),
    "classroom": ("in a classroom", True, ["a photo of a classroom"]),
    "store": ("in a shop", True, ["a photo of a shop aisle", "a photo of a supermarket"]),
    "warehouse": ("in a warehouse", True, ["a photo of a warehouse"]),
    "corridor": ("in a hallway", True, ["a photo of a corridor or hallway"]),
    "room": ("in a room", True, ["a photo of a room in a house", "a photo of an empty room"]),
    "hall": ("in a large hall", True, ["a photo of a large indoor hall"]),
    "car": ("in a car", True, ["a photo of the inside of a car"]),
    "street": ("on a street", False, ["a photo of a city street", "a photo of a sidewalk"]),
    "park": ("in a park", False, ["a photo of a park with grass and trees"]),
    "forest": ("in a forest", False, ["a photo of a forest"]),
    "plaza": ("in an open outdoor area", False,
              ["a photo of a public square or campus with paths and lawns",
               "a photo of a pedestrian area outside buildings"]),
    "parking lot": ("in a parking lot", False, ["a photo of a parking lot"]),
    "swimming pool": ("in a swimming pool", None,
                      ["a photo of a swimming pool", "a photo of people swimming in a pool"]),
    "dance studio": ("in a dance studio", True,
                     ["a photo of a dance studio", "a photo of a dance class"]),
    "stage": ("on a stage", None, ["a photo of a performance on a stage"]),
    "sports court": ("on a sports court", None,
                     ["a photo of a sports court", "a photo of a tennis or badminton court"]),
}

_INTERVAL_S = 2.0          # one look every couple of seconds: places don't move
_HISTORY = 4               # recent looks considered
_MIN_WINS = 3              # ...of which the place must win this many
_MIN_PROB = 0.55           # ...with this mean probability
_COARSE_MIN = 0.80         # indoor/outdoor fallback needs this much certainty
# Until there is enough history for the vote, two looks may name the place
# early — but only if they agree and clear a higher bar.
_EARLY_LOOKS = 2
_EARLY_MIN_PROB = 0.65
_WARMUP_INTERVAL_S = 0.5
# Once a place is shown it stays while it still averages this much and no
# other place has clearly taken over. Without this the name flipped between
# "in an office" and "indoors" whenever the vote sat near the bar.
_KEEP_PROB = 0.35


class SceneClassifier:
    """Background place recogniser. `available` is False if CLIP can't load."""

    def __init__(self, interval_s: float = _INTERVAL_S):
        self._interval = interval_s
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._history: deque[np.ndarray] = deque(maxlen=_HISTORY)
        self._phrase: str | None = None
        self._generation = 0
        self._model = None
        try:
            import torch
            from .clip_shared import get_clip
            self._torch = torch
            # Shared with the posture check (see clip_shared.py).
            self._model, self._proc, self._clip_lock = get_clip()
            self._names = list(_PLACES)
            from .clip_shared import encode_text
            rows = []
            for name in self._names:
                e = encode_text(_PLACES[name][2]).mean(0)
                rows.append(e / e.norm())
            self._text = torch.stack(rows)
            self._indoor = np.array([_PLACES[n][1] is True for n in self._names])
            self._outdoor = np.array([_PLACES[n][1] is False for n in self._names])
            self._phrase_index = {_PLACES[n][0]: i for i, n in enumerate(self._names)}
            log.info(f"scene description ready ({len(self._names)} places)")
        except Exception as e:
            log.warning(f"scene description unavailable, captions will omit the place ({e})")
            self._model = None
            return
        threading.Thread(target=self._loop, daemon=True).start()

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def phrase(self) -> str | None:
        with self._lock:
            return self._phrase

    def submit(self, frame: np.ndarray) -> None:
        """Offer the newest frame; only the latest one is ever looked at."""
        with self._lock:
            self._frame = frame

    def reset(self) -> None:
        with self._lock:
            self._frame = None
            self._history.clear()
            self._phrase = None
            self._generation += 1

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        import cv2
        while True:
            # Look quickly until the first answer is in (a new clip, a new
            # camera), then settle to the normal pace.
            with self._lock:
                warming = len(self._history) < _EARLY_LOOKS
            time.sleep(_WARMUP_INTERVAL_S if warming else self._interval)
            with self._lock:
                frame, self._frame = self._frame, None
                gen = self._generation
            if frame is None:
                continue
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                from .clip_shared import encode_images
                e = encode_images(rgb)
                probs = self._torch.softmax(100.0 * (e[0] @ self._text.T), -1).numpy()
            except Exception as ex:
                log.warning(f"scene look failed: {ex}")
                continue
            with self._lock:
                if gen != self._generation:      # source switched mid-look
                    continue
                self._history.append(probs)
                self._phrase = self._decide()

    def _decide(self) -> str | None:
        """Caller holds the lock."""
        if len(self._history) < _EARLY_LOOKS:
            return self._phrase
        hist = np.stack(self._history)
        mean = hist.mean(0)
        winners = hist.argmax(1)
        best = int(mean.argmax())
        if len(self._history) < _MIN_WINS:
            if (winners == best).all() and mean[best] >= _EARLY_MIN_PROB:
                return _PLACES[self._names[best]][0]
            return self._phrase
        if (winners == best).sum() >= _MIN_WINS and mean[best] >= _MIN_PROB:
            return _PLACES[self._names[best]][0]
        current = self._phrase_index.get(self._phrase)
        if current is not None and mean[current] >= _KEEP_PROB:
            return self._phrase
        if mean[self._indoor].sum() >= _COARSE_MIN:
            return "indoors"
        if mean[self._outdoor].sum() >= _COARSE_MIN:
            return "outdoors"
        return None
