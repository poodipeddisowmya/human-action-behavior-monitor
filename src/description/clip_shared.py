"""One CLIP model for the whole process.

Both the place recogniser (scene.py) and the posture check
(action_recognition/clip_posture.py) use the same zero-shot CLIP model.
Loading it once saves ~600 MB of memory and a few seconds of start-up; the
lock serialises the two users so they never run it at the same moment.

The model runs on the configured device (the GPU when there is one);
encode_text / encode_images return normalised features on the CPU so
callers never handle devices themselves.
"""
from __future__ import annotations

import os
import threading

CHECKPOINT = "openai/clip-vit-base-patch32"

_lock = threading.Lock()
_loaded: tuple | None = None
_failed = False


def get_clip():
    """(model, processor, lock), or raises if CLIP cannot be loaded here."""
    global _loaded, _failed
    with _lock:
        if _loaded is None and not _failed:
            try:
                # Weights must already be on disk: a monitor must not start
                # downloading, or fail, because it has no network.
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                from transformers import CLIPModel, CLIPProcessor
                from config import settings
                proc = CLIPProcessor.from_pretrained(CHECKPOINT)
                model = CLIPModel.from_pretrained(CHECKPOINT).eval().to(settings.device)
                _loaded = (model, proc, threading.Lock())
            except Exception:
                _failed = True
                raise
        if _loaded is None:
            raise RuntimeError("CLIP unavailable")
        return _loaded


def _features(out):
    # Newer transformers return a model-output object instead of a tensor.
    return out.pooler_output if hasattr(out, "pooler_output") else out


def encode_text(texts: list[str]):
    import torch
    model, proc, lock = get_clip()
    device = next(model.parameters()).device
    with lock, torch.no_grad():
        inputs = {k: v.to(device) for k, v in proc(text=texts, return_tensors="pt", padding=True).items()}
        e = _features(model.get_text_features(**inputs))
        return (e / e.norm(dim=-1, keepdim=True)).float().cpu()


def encode_images(images) -> "torch.Tensor":
    """images: one RGB array or a list of them."""
    import torch
    model, proc, lock = get_clip()
    device = next(model.parameters()).device
    with lock, torch.no_grad():
        inputs = {k: v.to(device) for k, v in proc(images=images, return_tensors="pt").items()}
        e = _features(model.get_image_features(**inputs))
        return (e / e.norm(dim=-1, keepdim=True)).float().cpu()
