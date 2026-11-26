"""Activities trained on your own clips: anything the base model has no
class for (asleep at a desk, a fight, someone lying in a corridor).

How it works: the activity model (clip_encoder.py) turns every 16-frame
look at a person into a feature vector, its summary of the clip. A trained
activity is a small layer on top of those vectors, learned from example
clips by scripts/train_activities.py (docs/TRAINING.md). Training takes
minutes on the machine that runs the monitor (Jetson, GPU or CPU PC); the activity model is not changed, so
nothing it already recognises is affected.

Every model also learns an "other" class from clips of ordinary activity
(desk work, walking, eating...): without it a trained layer can only ever
answer with one of its own activities, and would see them everywhere.

Model files: models/custom_activities/<backend>.npz (weights) and .json
(activity names, threshold, measured accuracy). Plain arrays and JSON, so
loading one never executes code. One file per backbone: features from S3D
and VideoMAE are not interchangeable, and the monitor loads the one that
matches its own backbone.

Live, a trained activity is shown once 3 of the last 4 looks at a person
(about 2 s) reach the activity's threshold (configurable), held for a few
seconds after the last supporting look, and it outranks the base model's
label. Nothing outranks a fall.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import settings

from .action_classifier import ActionPrediction

OTHER = "other"
FORMAT_VERSION = 1


def model_dir() -> Path:
    return settings.models_dir / "custom_activities"


def model_paths(backend: str) -> tuple[Path, Path]:
    base = model_dir() / backend
    return base.with_suffix(".npz"), base.with_suffix(".json")


# ---------------------------------------------------------------------------
# The trained layer
# ---------------------------------------------------------------------------
@dataclass
class ActivityHead:
    """Standardise the features, then a linear softmax layer."""
    backend: str
    classes: list[str]                 # includes OTHER
    mean: np.ndarray                   # (dim,)
    scale: np.ndarray                  # (dim,)
    weight: np.ndarray                 # (n_classes, dim)
    bias: np.ndarray                   # (n_classes,)
    threshold: float = 0.6
    meta: dict = field(default_factory=dict)

    @property
    def activities(self) -> list[str]:
        return [c for c in self.classes if c != OTHER]

    def probs(self, feats: np.ndarray) -> np.ndarray:
        """(n, dim) or (dim,) features -> class probabilities."""
        x = (np.atleast_2d(feats).astype(np.float32) - self.mean) / self.scale
        z = x @ self.weight.T + self.bias
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        return p[0] if np.ndim(feats) == 1 else p

    def save(self, npz: Path, meta_json: Path) -> None:
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, mean=self.mean, scale=self.scale, weight=self.weight, bias=self.bias)
        meta = {**self.meta, "format": FORMAT_VERSION, "backend": self.backend,
                "classes": self.classes, "threshold": self.threshold,
                "feature_dim": int(self.weight.shape[1])}
        meta_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, npz: Path, meta_json: Path) -> "ActivityHead":
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        if meta.get("format") != FORMAT_VERSION:
            raise ValueError(f"{meta_json.name}: unsupported format {meta.get('format')}")
        with np.load(npz, allow_pickle=False) as a:
            head = cls(backend=meta["backend"], classes=list(meta["classes"]),
                       mean=a["mean"].astype(np.float32), scale=a["scale"].astype(np.float32),
                       weight=a["weight"].astype(np.float32), bias=a["bias"].astype(np.float32),
                       threshold=float(meta.get("threshold", 0.6)), meta=meta)
        if head.weight.shape != (len(head.classes), head.mean.shape[0]):
            raise ValueError(f"{npz.name}: weights do not match {len(head.classes)} classes")
        return head


def train_head(features: np.ndarray, labels: np.ndarray, classes: list[str], backend: str,
               epochs: int = 300, weight_decay: float = 1e-3, seed: int = 0) -> ActivityHead:
    """Multinomial logistic regression on standardised features.

    Class-balanced (a class with few clips counts as much as one with many),
    with weight decay against overfitting a few hundred examples. Full-batch
    L-BFGS: deterministic, and converges in seconds on a CPU.
    """
    import torch
    torch.manual_seed(seed)
    x = features.astype(np.float32)
    mean = x.mean(axis=0)
    scale = x.std(axis=0) + 1e-6
    xt = torch.from_numpy((x - mean) / scale)
    yt = torch.from_numpy(labels.astype(np.int64))
    counts = np.bincount(labels, minlength=len(classes)).astype(np.float32)
    weights = torch.from_numpy(np.where(counts > 0, counts.sum() / (len(classes) * np.maximum(counts, 1)), 0.0)
                               .astype(np.float32))
    layer = torch.nn.Linear(x.shape[1], len(classes))
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.LBFGS(layer.parameters(), lr=0.5, max_iter=epochs, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = loss_fn(layer(xt), yt) + weight_decay * (layer.weight ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return ActivityHead(backend=backend, classes=list(classes), mean=mean, scale=scale,
                        weight=layer.weight.detach().numpy().copy(),
                        bias=layer.bias.detach().numpy().copy())


# ---------------------------------------------------------------------------
# Live use
# ---------------------------------------------------------------------------
def confirm_rule() -> tuple[int, int, float]:
    """(min_hits, looks, strong) from custom_activities: in model_config.yaml.

    A trained activity shows once `min_hits` of the last `looks` looks at a
    person reach its threshold. Shared with scripts/train_activities.py,
    which measures accuracy under exactly this rule.
    """
    cfg = dict(getattr(settings, "custom_activities", {}) or {})
    return int(cfg.get("min_hits", 3)), int(cfg.get("looks", 4)), float(cfg.get("strong", 0.98))


def strong_for(threshold: float, strong: float) -> float:
    """One look alone confirms only if it is beyond both `strong` and the
    threshold: a strict threshold must not be bypassed by a single look."""
    return max(strong, threshold + (1.0 - threshold) / 2)


@dataclass
class _Votes:
    looks: deque = field(default_factory=lambda: deque(maxlen=8))   # (activity, p) per look
    label: str | None = None
    conf: float = 0.0
    supported_at: float = 0.0


class CustomActivities:
    """The trained activities for one backbone, voted per person."""

    def __init__(self, head: ActivityHead):
        cfg = dict(getattr(settings, "custom_activities", {}) or {})
        self.head = head
        self.threshold = float(cfg.get("threshold") or head.threshold)
        self.min_hits, self.window, strong = confirm_rule()
        self.strong = strong_for(self.threshold, strong)
        self.hold_s = float(cfg.get("hold_s", 4.0))
        self._lock = threading.Lock()
        self._state: dict[int, _Votes] = {}

    @classmethod
    def load_for(cls, backend: str) -> "CustomActivities | None":
        """The trained model for this backbone, or None (none trained, or off)."""
        cfg = dict(getattr(settings, "custom_activities", {}) or {})
        if not cfg.get("enabled", True):
            return None
        npz, meta = model_paths(backend)
        if not (npz.is_file() and meta.is_file()):
            other = [b for b in ("s3d", "videomae") if b != backend and model_paths(b)[0].is_file()]
            if other:
                print(f"[custom] a model is trained for {other[0]}, not for {backend}: "
                      f"retrain with --backend {backend} to use it here", flush=True)
            return None
        try:
            head = ActivityHead.load(npz, meta)
        except Exception as e:
            print(f"[custom] {npz} not loaded: {e}", flush=True)
            return None
        print(f"[custom] trained activities ({backend}): {', '.join(head.activities)}", flush=True)
        return cls(head)

    @property
    def activities(self) -> list[str]:
        return self.head.activities

    def observe(self, key: int, feat: np.ndarray) -> None:
        """One look at a person (or the scene): called by the activity worker."""
        p = self.head.probs(feat)
        best = int(np.argmax(p))
        name, prob = self.head.classes[best], float(p[best])
        now = time.monotonic()
        with self._lock:
            st = self._state.setdefault(key, _Votes())
            st.looks.append((name, prob))
            if name == OTHER:
                return
            recent = list(st.looks)[-self.window:]
            hits = [q for n, q in recent if n == name and q >= self.threshold]
            if len(hits) >= self.min_hits or prob >= self.strong:
                st.label, st.conf, st.supported_at = name, float(np.mean(hits or [prob])), now
                print(f"[custom] {'scene' if key == 0 else f'P{key}'} {name} "
                      f"{len(hits)}/{len(recent)} looks, p={prob:.2f}", flush=True)
            elif st.label == name and prob >= self.threshold:
                st.supported_at = now

    def current(self, key: int) -> ActionPrediction | None:
        with self._lock:
            st = self._state.get(key)
            if st is None or st.label is None:
                return None
            if time.monotonic() - st.supported_at > self.hold_s:
                st.label = None
                return None
            return ActionPrediction(label=st.label, confidence=round(min(0.95, st.conf), 3))

    def forget(self, key: int) -> None:
        with self._lock:
            self._state.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._state.clear()


def describe(meta: dict) -> str:
    """One line for logs and the API: what was trained, when, how well."""
    acc = meta.get("validation", {}).get("clip_accuracy")
    when = meta.get("trained_at", "")[:10]
    return (f"{', '.join(c for c in meta.get('classes', []) if c != OTHER)}"
            f" ({meta.get('backend')}, trained {when}"
            + (f", {acc:.0%} of held-out clips correct" if acc is not None else "") + ")")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
