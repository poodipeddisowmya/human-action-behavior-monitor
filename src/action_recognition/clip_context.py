"""CLIP zero-shot classification — a fourth, genuinely different opinion.

Why this is not redundant with the S3D scene classifier
---------------------------------------------------------
S3D is trained on Kinetics-400: a FIXED, CLOSED vocabulary of 400 action
words, learned from short video clips, and therefore good at MOTION — a
push-up's up-down cycle, a bike's pedalling. Its coverage of our 13
categories is real but uneven and entirely dependent on what Kinetics happens
to have a word for (see target_vocabulary.py's audit comments — several
categories, e.g. "using phone", rest on a single matching class).

CLIP is trained on hundreds of millions of image-caption pairs from the open
web, with no fixed vocabulary at all — it scores an image against ANY text
you give it. That makes it strong at exactly what S3D is structurally weak
on: a single frame's SCENE and OBJECT semantics ("a person at a desk with a
laptop"), independent of motion. It has never seen "using phone" as a
training label, yet understands the phrase directly, because it learned
language and images jointly rather than a label lookup table.

Combining a motion-pretrained classifier with an image-text zero-shot one is
an established, well-documented pattern for exactly this kind of open-set
recognition — not a novel research bet.

Integration
------------
Same principle as object_context.py: feeds a vote into the EXISTING
consensus/hysteresis system in videomae_classifier.py rather than building a
second, competing decision path. A CLIP read that disagrees with what is
already showing has to out-evidence it exactly like any other vote.

Optional by design
--------------------
Needs `transformers`, already listed in requirements-optional.txt (pulled in
for the alternate VideoMAE backend) — no new dependency for anyone who
already has that installed. Degrades to `.available == False` otherwise,
exactly like every other optional layer in this codebase.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import settings

from .action_classifier import ActionPrediction

_CHECKPOINT = "openai/clip-vit-base-patch32"

# Natural-language description per category, fed through multiple prompt
# TEMPLATES below and averaged — "prompt ensembling", the standard technique
# from CLIP's own paper for squeezing extra zero-shot accuracy out of a fixed
# checkpoint with no fine-tuning. A single template is measurably noisier.
_CATEGORY_PHRASES: dict[str, str] = {
    "eating": "eating food",
    "drinking": "drinking a beverage",
    "cooking": "cooking food in a kitchen",
    "cleaning": "cleaning the house",
    "cycling": "riding a bicycle",
    "driving": "driving a car",
    "reading": "reading a book",
    "writing": "writing with a pen on paper",
    "using computer": "using a laptop computer",
    "using phone": "using a smartphone",
    "exercising": "exercising or working out",
    "smoking": "smoking a cigarette",
    "brushing teeth": "brushing their teeth",
}

_TEMPLATES = [
    "a photo of a person {}.",
    "a photo of someone {}.",
    "a person is {}.",
    "a candid photo of a person {}.",
]

# CLIP's raw softmax over a small closed set tends to be overconfident (the
# well-known zero-shot calibration issue) — this floor keeps a merely-plausible
# top guess out of the vote pool; the shared _tally() in videomae_classifier.py
# applies the real evidence bar on top of this.
_MIN_VOTE_CONF = 0.30
_MAX_VOTE_CONF = 0.85


class CLIPContextDetector:
    """Wraps a CLIP checkpoint for zero-shot image/category similarity."""

    def __init__(self, checkpoint: str = _CHECKPOINT):
        self._model = None
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._torch = torch
            self._processor = CLIPProcessor.from_pretrained(checkpoint)
            self._model = CLIPModel.from_pretrained(checkpoint).eval().to(settings.device)

            self._categories = list(_CATEGORY_PHRASES)
            prompts = [
                t.format(_CATEGORY_PHRASES[c])
                for c in self._categories
                for t in _TEMPLATES
            ]
            with torch.no_grad():
                text_inputs = self._processor(
                    text=prompts, return_tensors="pt", padding=True
                )
                text_inputs = {k: v.to(settings.device) for k, v in text_inputs.items()}
                text_features = self._model.get_text_features(**text_inputs)
                # transformers >=4.5x (verified: 5.0.0) changed get_*_features
                # to return a BaseModelOutputWithPooling instead of a bare
                # tensor by default — the projected embedding is its
                # `.pooler_output`. Older versions return the tensor directly.
                # Handling both keeps this working across the version range
                # already pinned in requirements-optional.txt without a hard
                # version pin of our own.
                if hasattr(text_features, "pooler_output"):
                    text_features = text_features.pooler_output
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Average the per-template embeddings back down to one per
            # category (prompt ensembling), then re-normalize.
            n_templates = len(_TEMPLATES)
            text_features = text_features.view(len(self._categories), n_templates, -1)
            text_features = text_features.mean(dim=1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            self._text_features = text_features

            print(f"[clip] loaded {checkpoint} — zero-shot over "
                  f"{len(self._categories)} categories, "
                  f"{n_templates} prompt templates each", flush=True)
        except Exception as e:
            print(f"[clip] unavailable, running without CLIP context. ({e})",
                  flush=True)
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    def infer(self, frame_bgr: np.ndarray) -> ActionPrediction | None:
        if self._model is None or frame_bgr is None or frame_bgr.size == 0:
            return None
        try:
            rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB, no copy needed for PIL
            from PIL import Image
            image = Image.fromarray(rgb)

            with self._torch.no_grad():
                img_inputs = self._processor(images=image, return_tensors="pt")
                img_inputs = {k: v.to(settings.device) for k, v in img_inputs.items()}
                image_features = self._model.get_image_features(**img_inputs)
                if hasattr(image_features, "pooler_output"):
                    image_features = image_features.pooler_output
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # CLIP's own learned logit scale, same as its standard
                # zero-shot classification recipe — not an arbitrary choice.
                logit_scale = self._model.logit_scale.exp()
                logits = logit_scale * image_features @ self._text_features.T
                probs = self._torch.softmax(logits, dim=-1)[0]

            best_idx = int(self._torch.argmax(probs))
            confidence = float(probs[best_idx])
        except Exception as e:
            print(f"[clip] inference error: {e}", flush=True)
            return None

        if confidence < _MIN_VOTE_CONF:
            return None
        category = self._categories[best_idx]
        return ActionPrediction(label=category, confidence=min(_MAX_VOTE_CONF, confidence))
