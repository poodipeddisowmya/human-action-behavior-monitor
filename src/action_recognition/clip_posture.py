"""Sitting or standing, from how the person LOOKS rather than their joints.

The pose rules decide "sitting" from the knees: knees level with or above
the hips. That works when the legs are in view and fails in the most common
live-camera framing of all — a person at a desk facing a webcam, cut off at
the waist. With no knees the rules fall back to motion, and a still person
then read as "standing". Measured on 19 clips with a known posture (237
frames): the pose rules were right on 58% of frames, and on 0% for people
seated at a desk (demo-using-computer, classroom-3); zero-shot CLIP, shown
the frame and asked "sitting or standing?", was right on 86%, including 14
of 14 on both of those.

So when the pose engine has nothing better than stillness to go on, this
asks CLIP. It runs on its own thread about once a second per person over the
newest image only; update() never waits for it.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from src.utils import get_logger

log = get_logger(__name__)

_PROMPTS = {
    "sitting": ["a photo of a person sitting", "a photo of a person sitting on a chair",
                "a photo of a person sitting at a desk", "a photo of a person sitting on a sofa",
                "a photo of a person sitting on the ground"],
    "standing": ["a photo of a person standing", "a photo of a person standing up straight",
                 "a photo of a person walking"],
}
_INTERVAL_S = 1.0      # one look per person per second
_MAX_AGE_S = 3.0       # an answer older than this is not used
_SURE = 0.75           # probability needed to say either posture
_MAX_PEOPLE = 4        # looks per cycle, so a crowd cannot starve the CPU


class ClipPosture:
    def __init__(self):
        self._lock = threading.Lock()
        self._pending: dict[int, np.ndarray] = {}
        self._result: dict[int, tuple[float, float]] = {}   # key -> (p_sitting, time)
        self._asked_at: dict[int, float] = {}
        self._generation = 0
        try:
            import torch
            from src.description.clip_shared import get_clip
            self._torch = torch
            self._model, self._proc, self._clip_lock = get_clip()
            labels, texts = [], []
            for k, v in _PROMPTS.items():
                labels += [k] * len(v)
                texts += v
            self._sit = np.array([l == "sitting" for l in labels])
            from src.description.clip_shared import encode_text
            self._text = encode_text(texts)
            self.available = True
        except Exception as e:
            log.warning(f"posture check unavailable ({e})")
            self.available = False
            return
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, key: int, image: np.ndarray) -> None:
        """Offer the newest image of this person; ignored if asked recently."""
        if not self.available or image is None or image.size == 0:
            return
        now = time.time()
        with self._lock:
            if now - self._asked_at.get(key, 0.0) < _INTERVAL_S:
                return
            self._asked_at[key] = now
            self._pending[key] = image

    def posture(self, key: int) -> str | None:
        """'sitting' / 'standing' when recently and clearly decided, else None."""
        with self._lock:
            r = self._result.get(key)
        if r is None or time.time() - r[1] > _MAX_AGE_S:
            return None
        if r[0] >= _SURE:
            return "sitting"
        if r[0] <= 1 - _SURE:
            return "standing"
        return None

    def forget(self, key: int) -> None:
        with self._lock:
            for d in (self._pending, self._result, self._asked_at):
                d.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._pending.clear()
            self._result.clear()
            self._asked_at.clear()
            self._generation += 1

    def _loop(self) -> None:
        import cv2
        while True:
            time.sleep(0.1)
            with self._lock:
                if not self._pending:
                    continue
                batch = list(self._pending.items())[:_MAX_PEOPLE]
                for k, _ in batch:
                    self._pending.pop(k, None)
                gen = self._generation
            try:
                imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for _, img in batch]
                from src.description.clip_shared import encode_images
                e = encode_images(imgs)
                p = self._torch.softmax(100.0 * (e @ self._text.T), -1).numpy()
            except Exception as ex:
                log.warning(f"posture look failed: {ex}")
                continue
            now = time.time()
            with self._lock:
                if gen != self._generation:
                    continue
                for (k, _), row in zip(batch, p):
                    self._result[k] = (float(row[self._sit].sum()), now)
