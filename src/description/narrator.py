"""Detailed, scene-aware subtitles from a small vision-language model (GPU).

The template captions (captions.py) say what the action labels say:
"A person is cooking in a kitchen." This narrator looks at the video itself
and writes a factual sentence or two about the people, what they are doing
and the surroundings, e.g. "A person is eating a sandwich with avocado and
eggs." It is grounded: the prompt carries what the recognition system
measured (how many people, the main activity, the place), so the model
describes the scene instead of inventing one.

GPU only. SmolVLM2-500M takes 75-115 s per description on a laptop CPU and
about a second on a T4. Off by default; the GPU profile turns it on
(config/model_config.yaml, vlm_captions). If the model is not on disk, or
fails, captions fall back to the template ones without any other effect.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque

import cv2
import numpy as np

from src.utils import get_logger

log = get_logger(__name__)

CHECKPOINT = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
_INTERVAL_S = 4.0          # one description every few seconds
_FRAMES = 3                # frames per description, spread over...
_SPAN_S = 2.0              # ...the last two seconds
_MAX_SIDE = 512            # frame size given to the model
_FRESH_S = 12.0            # a description older than this is not shown
_MAX_WORDS = 25


_PERSON_N = re.compile(r"\b[Pp]erson\s?(\d+)\b")
# Gender is what the model got wrong most visibly on a T4 (a woman dancer
# described as "a man"), so it is never stated: rewritten after generation,
# which is reliable, rather than forbidden in the prompt, which made the
# 500M model collapse to "A person." when tested.
_PLURAL = {"men", "women", "boys", "girls", "ladies", "guys"}
_GENDERED = re.compile(r"\b(man|woman|boy|girl|lady|guy|men|women|boys|girls|ladies|guys)\b",
                       re.I)
_PRONOUNS = {"he": "they", "she": "they", "his": "their", "her": "their",
             "him": "them", "himself": "themself", "herself": "themself"}
_PRONOUN = re.compile(r"\b(he|she|his|her|him|himself|herself)\b", re.I)


def _people_words(text: str) -> str:
    """'Person1 ... person2' -> 'A person ... another person'; no gender words."""
    def name(m: re.Match) -> str:
        return {"1": "a person", "2": "another person"}.get(m.group(1), "someone")
    text = _PERSON_N.sub(name, text)
    text = _GENDERED.sub(lambda m: "people" if m.group(1).lower() in _PLURAL else "person", text)
    text = _PRONOUN.sub(lambda m: _PRONOUNS[m.group(1).lower()], text)
    plural_verb = {"is": "are", "was": "were", "has": "have"}
    text = re.sub(r"\b(they) (is|was|has)\b",
                  lambda m: f"{m.group(1)} {plural_verb[m.group(2).lower()]}", text, flags=re.I)
    text = re.sub(r"\ba person and a person\b", "two people", text, flags=re.I)
    # Re-capitalise sentence starts the substitution may have lowered.
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def _sentence(text: str) -> str:
    """Tidy model output into one or two clean sentences."""
    text = " ".join(text.replace("\n", " ").split()).strip(" \"'")
    text = _people_words(text)
    words = text.split()
    if len(words) > _MAX_WORDS:                        # cut at the last full stop
        text = " ".join(words[:_MAX_WORDS])
        text = text[: text.rfind(".") + 1] if "." in text else text + "."
    if text and text[-1] not in ".!?":
        text += "."
    return text[:1].upper() + text[1:]


def build_prompt(people: int, activity: str | None, place: str | None) -> str:
    """Short, positive, grounded in what was measured. A long list of
    prohibitions made the 500M model answer just "A person." (tested), so the
    count is stated as fact and gender is handled in _people_words()."""
    facts = ["There is exactly one person in view" if people == 1
             else f"There are {people} people in view"]
    if activity:
        facts.append(f"the main activity is {activity}")
    if place:                                  # "in a kitchen" -> "a kitchen"
        facts.append("the place is " + re.sub(r"^(in|at|on) ", "", place))
    subject = "the person is" if people == 1 else "the people are"
    return ("; ".join(facts) + ". Describe this video in one detailed sentence: what "
            f"{subject} doing, any object being used, and the surroundings.")


class SceneNarrator:
    def __init__(self, device: str):
        self._device = device
        self._lock = threading.Lock()
        self._frames: deque[tuple[float, np.ndarray]] = deque(maxlen=64)
        self._context: tuple[int, str | None, str | None] = (0, None, None)
        self._text: str | None = None
        self._text_at = 0.0
        self._generation = 0
        self.available = False
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self._torch = torch
            self._proc = AutoProcessor.from_pretrained(CHECKPOINT)
            dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
            self._model = AutoModelForImageTextToText.from_pretrained(
                CHECKPOINT, torch_dtype=dtype).eval().to(device)
            self.available = True
            threading.Thread(target=self._loop, daemon=True).start()
            log.info(f"scene narrator ready ({CHECKPOINT} on {device})")
        except Exception as e:
            log.warning(f"scene narrator unavailable, using template captions ({e})")

    # -------- called from the pipeline --------
    def submit(self, frame: np.ndarray, people: int, activity: str | None,
               place: str | None) -> None:
        # Shrunk on arrival: 64 full HD frames would hold ~170 MB.
        h, w = frame.shape[:2]
        s = min(1.0, _MAX_SIDE / max(h, w))
        small = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s)))) if s < 1 else frame.copy()
        with self._lock:
            self._frames.append((time.monotonic(), small))
            self._context = (people, activity, place)

    @property
    def text(self) -> str | None:
        """The latest description, if it is recent enough to still be true."""
        with self._lock:
            if self._text and time.monotonic() - self._text_at <= _FRESH_S:
                return self._text
            return None

    def reset(self) -> None:
        with self._lock:
            self._frames.clear()
            self._text = None
            self._generation += 1

    # -------- background worker --------
    def _pick(self) -> tuple[list, tuple, int] | None:
        with self._lock:
            if len(self._frames) < 2:
                return None
            now = self._frames[-1][0]
            recent = [f for t, f in self._frames if now - t <= _SPAN_S]
            gen, ctx = self._generation, self._context
        idx = np.linspace(0, len(recent) - 1, min(_FRAMES, len(recent))).round().astype(int)
        return [recent[i] for i in idx], ctx, gen

    def _describe(self, frames: list[np.ndarray], ctx: tuple) -> str:
        from PIL import Image
        imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": build_prompt(*ctx)}]
        prompt = self._proc.apply_chat_template([{"role": "user", "content": content}],
                                                add_generation_prompt=True)
        inputs = self._proc(text=prompt, images=imgs, return_tensors="pt").to(self._device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)
        with self._torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=64, do_sample=False)
        return _sentence(self._proc.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])

    def _loop(self) -> None:
        while True:
            t0 = time.monotonic()
            picked = self._pick()
            if picked:
                frames, ctx, gen = picked
                try:
                    text = self._describe(frames, ctx)
                except Exception as e:
                    log.warning(f"scene narration failed: {e}")
                    text = None
                with self._lock:
                    if text and gen == self._generation:      # not a stale source
                        self._text, self._text_at = text, time.monotonic()
            time.sleep(max(0.5, _INTERVAL_S - (time.monotonic() - t0)))
