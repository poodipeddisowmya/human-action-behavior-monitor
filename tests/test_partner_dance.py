"""A partner dance needs two people (videomae_classifier._PARTNER_ONLY).

Measured on a T4: a lone runner read as "tango dancing" and showed "dancing"
37% of the time once dancing became a target activity.
"""
from __future__ import annotations

import numpy as np
import torch
from torchvision.models.video import S3D_Weights

from src.action_recognition.target_vocabulary import resolve_indices
from src.action_recognition.videomae_classifier import VideoMAEClassifier

CATS = S3D_Weights.KINETICS400_V1.meta["categories"]


def _classifier_answering(raw: str) -> VideoMAEClassifier:
    """The real _infer path over a stand-in model that always says `raw`."""
    c = VideoMAEClassifier.__new__(VideoMAEClassifier)
    c._torch, c._backend = torch, "s3d"
    c._id2label = dict(enumerate(CATS))
    c._allowed, c._idx_to_cat = resolve_indices(c._id2label)
    c._allowed_t = torch.tensor(c._allowed)
    c._mean = torch.zeros(1, 3, 1, 1, 1)
    c._std = torch.ones(1, 3, 1, 1, 1)
    logits = torch.full((1, len(CATS)), -10.0)
    logits[0, CATS.index(raw)] = 10.0
    c._model = lambda x: logits
    return c


CLIP = [np.zeros((224, 224, 3), np.uint8)] * 16


def test_partner_dance_rejected_with_one_person():
    assert _classifier_answering("tango dancing")._infer(CLIP, solo=True) == (None, None)


def test_partner_dance_accepted_with_two_people():
    cat, conf = _classifier_answering("tango dancing")._infer(CLIP, solo=False)
    assert cat == "dancing" and conf > 0.5


def test_solo_dance_styles_unaffected():
    cat, _ = _classifier_answering("dancing ballet")._infer(CLIP, solo=True)
    assert cat == "dancing"


def test_swimming_resolves_to_its_own_category():
    cat, _ = _classifier_answering("swimming backstroke")._infer(CLIP, solo=True)
    assert cat == "swimming"
