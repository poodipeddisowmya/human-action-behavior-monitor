"""Open-vocabulary activities: recognised from a text description.

The activity models (VideoMAE / S3D) only know Kinetics-400's classes.
X-CLIP is a video transformer trained to match video with text, so it can
score a look against activities it was never trained on ("taking tablets").
Activities are configured in model_config.yaml (open_vocab:) as a name and a
few descriptions.

Every look is scored against the configured activities AND a background set
(the activities the system already tracks, and plain postures). Softmax over
all of them is what keeps a look of someone drinking water from scoring as
"taking tablets": the drinking description simply wins it.

A label is confirmed when min_hits of the last 3 looks of the same person
(or scene) reach the threshold, and is shown until hold_s passes with no
supporting look. Measured numbers are in model_config.yaml.

Runs in its own thread, about one look a second: the video and the other
models never wait for it. GPU only (a look takes seconds on a CPU).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import settings
from config.settings import is_jetson

from .action_classifier import ActionPrediction

log = logging.getLogger(__name__)

# What a look can be instead of a configured activity. Descriptions as
# measured with scripts/eval_open_vocab.py (same prompts, same scores).
BACKGROUND: dict[str, list[str]] = {
    "brushing teeth": ["brushing teeth", "a person brushing their teeth with a toothbrush"],
    "cleaning": ["cleaning the floor", "a person mopping or cleaning"],
    "cooking": ["cooking food in a kitchen", "a person cooking on a stove"],
    "cycling": ["riding a bicycle", "a person cycling"],
    "dancing": ["dancing", "a person dancing"],
    "drinking": ["drinking from a cup or bottle", "a person drinking a beverage"],
    "eating": ["eating food", "a person eating a meal"],
    "exercising": ["exercising", "a person doing push ups, squats or pull ups"],
    "reading": ["reading a book", "a person reading"],
    "smoking": ["smoking a cigarette", "a person smoking"],
    "swimming": ["swimming", "a person swimming in a pool"],
    "using computer": ["typing on a computer keyboard", "a person using a laptop"],
    "using phone": ["using a mobile phone", "a person looking at a smartphone"],
    "writing": ["writing with a pen", "a person writing on paper"],
    "walking": ["walking", "a person walking"],
    "sitting": ["sitting", "a person sitting on a chair"],
    "lying down": ["lying down", "a person lying on a bed or sofa"],
    "falling down": ["falling down", "a person falling to the floor"],
    # Office scenes. Without these, the nearest description won: two
    # colleagues chatting over mugs in a pantry read "making tea" 71% of
    # the time, and a desk-phone call "smoking" (Pexels RDNE office clips,
    # 2026-11-23).
    "talking": ["people talking to each other", "two people having a conversation holding mugs"],
    "phone call": ["talking on the phone", "a person holding a telephone to their ear"],
}

# The activity model's known confusions, and how X-CLIP settles them. A
# handset at the ear and a cigarette look alike to it; so do sipping from a
# mug and taking a bite, and writing on paper at a desk and typing (the
# monitor in view decides it). When X-CLIP's recent looks at the same person
# clearly favour another reading, that is the one shown:
#   shown label -> {label instead: X-CLIP descriptions that support it}
CROSS_CHECK: dict[str, dict[str, tuple[str, ...]]] = {
    "smoking": {"using phone": ("phone call", "using phone")},
    "eating": {"drinking": ("drinking", "talking")},
    "using computer": {"writing": ("writing",), "reading": ("reading",)},
}
_CROSS_LOOKS_S = 6.0     # looks this recent count
_CROSS_MARGIN = 1.5      # the other reading must score this many times higher

_SPAN_S = 2.5           # a look covers this much video (as measured)
_SHORT_SIDE = 320       # frames are kept at this size until a look
_PAD = 0.6              # context around a person's box, multi-person scenes
_SCENE = 0              # key for "the whole frame" (one person in view)
_LOAD_RETRY_S = 20.0    # retries after a failed first load, freeing memory between
_LOAD_ATTEMPTS = 4


@dataclass
class _State:
    frames: deque = field(default_factory=lambda: deque(maxlen=128))   # (t, rgb)
    last_frame_at: float = 0.0
    last_look_at: float = 0.0
    scores: deque = field(default_factory=lambda: deque(maxlen=3))     # {activity: p}
    looks: deque = field(default_factory=lambda: deque(maxlen=4))      # (t, {every name: p})
    label: str | None = None
    conf: float = 0.0
    supported_at: float = 0.0


class OpenVocabRecognizer:
    def __init__(self) -> None:
        cfg = settings.open_vocab or {}
        self.activities: dict[str, list[str]] = {
            str(k): [str(p) for p in v] for k, v in (cfg.get("activities") or {}).items() if v}
        self.enabled = bool(cfg.get("enabled", False)) and bool(self.activities) \
            and settings.device.startswith("cuda")
        self.interval_s = float(cfg.get("interval_s", 1.0))
        self.threshold = float(cfg.get("threshold", 0.35))
        # Per-activity bars where the measured margin allows (see the config).
        self.thresholds = {a: float((cfg.get("thresholds") or {}).get(a, self.threshold))
                           for a in self.activities}
        self.min_hits = int(cfg.get("min_hits", 2))
        # One look this strong confirms on its own (see the config).
        self.strong = float(cfg.get("strong", 1.01))
        self.hold_s = float(cfg.get("hold_s", 4.0))
        self._states: dict[int, _State] = {}
        self._lock = threading.Lock()
        self._model = None
        self._jobs: queue.Queue = queue.Queue(maxsize=1)
        self._ready = False
        self._checkpoint = str(cfg.get("model", "microsoft/xclip-base-patch16-zero-shot"))

    def start(self) -> None:
        """Load the model (the pipeline's warm-up calls this last, after the
        other models). One attempt now, so the first frame already has it;
        if that fails for memory (a desktop session on an 8 GB Jetson), keep
        retrying in the background and let the pipeline start without it."""
        if not self.enabled or self._ready:
            return
        try:
            self._load(self._checkpoint)
        except Exception as e:
            self._model = None
            print(f"[open-vocab] load failed ({e}); retrying in the background", flush=True)
            threading.Thread(target=self._start, args=(self._checkpoint,),
                             daemon=True, name="open-vocab").start()
            return
        self._ready = True
        print(f"[open-vocab] {list(self.activities)} via {self._checkpoint} on {settings.device}",
              flush=True)
        threading.Thread(target=self._worker, daemon=True, name="open-vocab").start()

    def _start(self, checkpoint: str) -> None:
        for attempt in range(2, _LOAD_ATTEMPTS + 1):
            time.sleep(_LOAD_RETRY_S)
            try:
                self._load(checkpoint)
                break
            except Exception as e:
                self._model = None
                try:
                    self._torch.cuda.empty_cache()
                except Exception:
                    pass
                print(f"[open-vocab] load attempt {attempt}/{_LOAD_ATTEMPTS} failed ({e})", flush=True)
        else:
            self.enabled = False
            print(f"[open-vocab] unavailable; configured activities {list(self.activities)} "
                  "will not be recognised (GPU memory: close other GPU apps, or run headless)",
                  flush=True)
            return
        self._ready = True
        print(f"[open-vocab] {list(self.activities)} via {checkpoint} on {settings.device}",
              flush=True)
        self._worker()

    # ------------------------------------------------------------------
    def _load(self, checkpoint: str) -> None:
        import torch
        from transformers import XCLIPModel, XCLIPProcessor

        self._torch = torch
        self._release = is_jetson()
        if is_jetson():
            # It loads last, after ~1.5 GB of other weights were read into the
            # page cache, which CUDA allocations cannot evict on a Jetson
            # (NvMapMemAlloc error 12). Touching a buffer evicts it.
            np.ones(int(1.5 * 2**30) // 8).sum()
        self._proc = XCLIPProcessor.from_pretrained(checkpoint)
        self._model = XCLIPModel.from_pretrained(
            checkpoint, torch_dtype=torch.float16).eval().to(settings.device)
        self._nf = self._model.config.vision_config.num_frames
        # Configured activities first, then background ones not configured.
        self._names = list(self.activities) + [n for n in BACKGROUND if n not in self.activities]
        prompts = {**BACKGROUND, **self.activities}
        self._k = max(len(prompts[n]) for n in self._names)
        texts = []
        for n in self._names:                      # pad to k prompts each (repeat)
            ps = prompts[n]
            texts += [ps[i % len(ps)] for i in range(self._k)]
        t = self._proc(text=texts, return_tensors="pt", padding=True)
        m = self._model
        # The descriptions never change, so they are encoded once and X-CLIP's
        # text encoder is then dropped: ~126 MB less GPU memory on an 8 GB
        # Jetson, and no 40-prompt text pass on every look. Only the part of
        # the text side that depends on the video (the prompt generator) runs
        # per look; the scores are the full model's (checked to fp16 noise).
        with torch.no_grad():
            out = m.text_model(**{k: v.to(settings.device) for k, v in t.items()})
            self._text_embeds = m.text_projection(out[1])
        m.text_model = None
        torch.cuda.empty_cache()
        # Warm-up: the first look also compiles kernels.
        dummy = [np.zeros((224, 224, 3), np.uint8)] * self._nf
        self._score(dummy)

    def _score(self, clip: list[np.ndarray]) -> dict[str, float]:
        """X-CLIP's forward pass with the text side precomputed (see _load)."""
        torch, m = self._torch, self._model
        v = self._proc(videos=[clip], return_tensors="pt")["pixel_values"]
        v = v.to(settings.device, torch.float16)
        with torch.no_grad():
            b, n, c, h, w = v.shape
            vis = m.vision_model(pixel_values=v.reshape(-1, c, h, w))
            video = m.visual_projection(vis[1])
            video = m.mit(video.view(b, n, -1))[1]
            img = m.prompts_visual_layernorm(vis[0][:, 1:, :]) @ m.prompts_visual_projection
            img = img.view(b, n, -1, video.shape[-1]).mean(dim=1)
            text = self._text_embeds.unsqueeze(0).expand(b, -1, -1)
            text = text + m.prompts_generator(text, img)
            video = video / video.norm(p=2, dim=-1, keepdim=True)
            text = text / text.norm(p=2, dim=-1, keepdim=True)
            logits = torch.einsum("bd,bkd->bk", video, m.logit_scale.exp() * text)[0].float()
        p = torch.softmax(logits.view(len(self._names), self._k).mean(1), -1).cpu().numpy()
        del v, vis, video, img, text, logits
        if self._release:
            # A look's working memory would otherwise stay reserved by
            # PyTorch, where the TensorRT person detector cannot use it: on an
            # 8 GB Jetson that ran the detector out of memory. A few ms a look.
            torch.cuda.empty_cache()
        return {n: float(p[i]) for i, n in enumerate(self._names)}

    # ------------------------------------------------------------------
    def observe(self, person_id: int, frame: np.ndarray, xyxy, single_person: bool) -> None:
        """Called per tracked person per processed frame; cheap when not sampling."""
        if not self._ready:
            return
        key = _SCENE if single_person else person_id
        now = time.monotonic()
        step = _SPAN_S / self._nf
        with self._lock:
            st = self._states.setdefault(key, _State())
            if now - st.last_frame_at < step:
                return
            st.last_frame_at = now
        img = frame
        if not single_person:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            bw, bh = x2 - x1, y2 - y1
            img = frame[max(0, int(y1 - bh * _PAD)):min(h, int(y2 + bh * _PAD)),
                        max(0, int(x1 - bw * _PAD)):min(w, int(x2 + bw * _PAD))]
            if img.size == 0:
                return
        s = _SHORT_SIDE / min(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)),
                             interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with self._lock:
            st.frames.append((now, rgb))
        # Offer a look if this key is due; the worker takes the newest offer.
        if now - st.last_look_at >= self.interval_s and len(st.frames) >= self._nf:
            clip = self._clip(st, now)
            if clip is not None:
                try:
                    self._jobs.put_nowait((key, clip))
                    st.last_look_at = now
                except queue.Full:
                    pass

    def _clip(self, st: _State, now: float) -> list[np.ndarray] | None:
        with self._lock:
            frames = [f for f in st.frames if now - f[0] <= _SPAN_S]
        if len(frames) < self._nf // 2:
            return None
        idx = np.linspace(0, len(frames) - 1, self._nf).astype(int)
        return [frames[i][1] for i in idx]

    def _worker(self) -> None:
        while True:
            key, clip = self._jobs.get()
            scores = None
            for attempt in (1, 2):
                try:
                    scores = self._score(clip)
                    break
                except Exception as e:
                    # On a shared-memory Jetson a look can fail for memory
                    # while every model makes its first allocations; freeing
                    # PyTorch's cache and trying once more recovers it.
                    if attempt == 2:
                        print(f"[open-vocab] look failed: {e}", flush=True)
                    self._torch.cuda.empty_cache()
                    time.sleep(0.2)
            if scores is not None:
                self._vote(key, scores)

    def _vote(self, key: int, scores: dict[str, float]) -> None:
        now = time.monotonic()
        with self._lock:
            st = self._states.get(key)
            if st is None:
                return
            st.scores.append({a: scores[a] for a in self.activities})
            st.looks.append((now, scores))
            best = max(self.activities, key=lambda a: scores[a])
            top = max(scores, key=scores.get)
            print(f"[open-vocab] look {'scene' if key == _SCENE else f'P{key}'}: {best} {scores[best]:.2f}"
                  f" (top: {top} {scores[top]:.2f})", flush=True)
            bar = self.thresholds.get(best, self.threshold)
            hits = [s[best] for s in st.scores if s[best] >= bar]
            if len(hits) >= self.min_hits or scores[best] >= self.strong:
                st.label, st.conf, st.supported_at = best, float(np.mean(hits)), now
                print(f"[open-vocab] {'scene' if key == _SCENE else f'P{key}'} {best} "
                      f"{len(hits)}/{len(st.scores)} looks, mean {st.conf:.2f}", flush=True)
            elif st.label is not None and scores[st.label] >= self.thresholds.get(st.label, self.threshold):
                st.supported_at = now

    # ------------------------------------------------------------------
    def current(self, person_id: int, single_person: bool) -> ActionPrediction | None:
        """The confirmed open-vocabulary activity for this person, if any."""
        if not self._ready:
            return None
        with self._lock:
            st = self._states.get(_SCENE if single_person else person_id)
            if st is None or st.label is None:
                return None
            if time.monotonic() - st.supported_at > self.hold_s:
                st.label = None
                return None
            # Scores of 0.35-0.9 are shares among ~20 descriptions; shown on
            # the activities' usual 0.5-0.95 scale.
            return ActionPrediction(label=st.label, confidence=round(min(0.95, 0.5 + st.conf / 2), 3))

    def cross_check(self, person_id: int, single_person: bool, label: str) -> str:
        """The label to show instead of `label`, if X-CLIP clearly disagrees
        in a known confusion (CROSS_CHECK); else `label` unchanged."""
        rule = CROSS_CHECK.get(label)
        if rule is None or not self._ready:
            return label
        now = time.monotonic()
        with self._lock:
            st = self._states.get(_SCENE if single_person else person_id)
            looks = [s for t, s in st.looks if now - t <= _CROSS_LOOKS_S] if st else []
        if not looks:
            return label
        same = float(np.mean([s.get(label, 0.0) for s in looks]))
        best, best_score = label, _CROSS_MARGIN * same
        for instead, support in rule.items():
            other = float(np.mean([max(s.get(n, 0.0) for n in support) for s in looks]))
            if other >= best_score:
                best, best_score = instead, other
        return best

    def forget(self, person_id: int) -> None:
        with self._lock:
            self._states.pop(person_id, None)

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
