"""The activity model itself: a 16-frame clip in, Kinetics-400 scores and a
feature vector out.

Shared by the live classifier (videomae_classifier.py) and the training
script (scripts/train_activities.py), so a clip used to train a new activity
goes through exactly the model, preprocessing and precision the monitor
uses live. The feature vector is the model's summary of the clip, taken
just before its own 400-way output layer; trained activities
(custom_activities.py) are a small layer on top of it.

Two backbones, both pretrained on Kinetics-400:

    's3d'       torchvision S3D: 8.3M parameters, 1024-dim features,
                BSD licence (commercial use allowed), fast on a CPU
    'videomae'  VideoMAE-base: 86.5M parameters, 768-dim features, more
                accurate on a GPU; weights CC BY-NC 4.0 (non-commercial)
"""
from __future__ import annotations

import cv2
import numpy as np

from config import settings

CLIP_LEN = 16
INPUT_HW = (224, 224)
_KIN_MEAN = (0.43216, 0.394666, 0.37645)
_KIN_STD = (0.22803, 0.22145, 0.216989)
VMAE_CHECKPOINT = "MCG-NJU/videomae-base-finetuned-kinetics"
BACKENDS = ("s3d", "videomae")


class ClipEncoder:
    """Loads one backbone; `forward(clip)` -> (logits, features)."""

    def __init__(self, backend: str | None = None, device: str | None = None):
        import torch
        self._torch = torch
        self.device = device or settings.device
        self.backend = (backend or getattr(settings.model, "deep_backend", "s3d")).lower()
        if self.backend not in BACKENDS:
            raise ValueError(f"unknown backend {self.backend!r}; use one of {BACKENDS}")
        self.processor = None
        # On a CPU, the model compiled for this machine by
        # scripts/optimise_for_device.py when there is one (same outputs,
        # several times faster; src/utils/compiled_models.py).
        self._compiled = None
        self.runtime = "pytorch"
        if self.backend == "videomae":
            self.model, self.id2label = self._load_videomae()
            self.dim = int(self.model.config.hidden_size)
        else:
            self.model, self.id2label = self._load_s3d()
            self.dim = int(self.model.classifier[1].in_channels)
            if str(self.device) == "cpu":
                from src.utils.compiled_models import activity_model
                found = activity_model("s3d")
                if found:
                    self._compiled, self.runtime = found

    # ---- loaders ----
    def _load_s3d(self):
        from torchvision.models.video import S3D_Weights, s3d
        weights = S3D_Weights.KINETICS400_V1
        model = s3d(weights=weights).eval().to(self.device)
        t = self._torch
        self._mean = t.tensor(_KIN_MEAN, device=self.device).view(1, 3, 1, 1, 1)
        self._std = t.tensor(_KIN_STD, device=self.device).view(1, 3, 1, 1, 1)
        return model, dict(enumerate(weights.meta["categories"]))

    def _load_videomae(self):
        try:
            from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
        except ImportError as e:
            raise ImportError(
                "deep_backend 'videomae' needs the optional extras: "
                "pip install -r requirements-optional.txt (or use deep_backend: s3d)") from e
        self.processor = VideoMAEImageProcessor.from_pretrained(VMAE_CHECKPOINT)
        model = VideoMAEForVideoClassification.from_pretrained(VMAE_CHECKPOINT)
        model.eval().to(self.device)
        if getattr(settings.model, "deep_half", False) and str(self.device).startswith("cuda"):
            # Jetson profile: 2.7x faster per look on an Orin Nano (202 -> 75
            # ms). Softmax still runs in FP32, so the calibrated coverage /
            # confidence thresholds see the same numbers.
            model.half()
        t = self._torch
        self._vmae_mean = t.tensor(self.processor.image_mean, device=self.device).view(1, 1, 3, 1, 1)
        self._vmae_std = t.tensor(self.processor.image_std, device=self.device).view(1, 1, 3, 1, 1)
        return model, model.config.id2label

    # ---- inference ----
    def forward(self, clip_bgr: list[np.ndarray]):
        """(logits: float32 tensor [400], features: float32 numpy [dim])."""
        torch = self._torch
        rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in clip_bgr], axis=0)
        with torch.no_grad():
            if self.backend == "s3d":
                # (T,H,W,C) uint8 -> (1,C,T,H,W) normalised float. Contiguous,
                # so the call matches the warm-up's memory layout: with cuDNN
                # autotuning a permuted input is a "new" shape and its first
                # call paid ~4 s of tuning on a T4.
                x = torch.from_numpy(rgb).to(self.device).float().div_(255.0)
                x = x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
                x = (x - self._mean) / self._std
                if self._compiled is not None:
                    logits, feat = self._compiled(x.numpy())
                    return torch.from_numpy(logits[0]), feat[0]
                m = self.model
                pooled = m.avgpool(m.features(x))                    # (1,1024,t,h,w)
                # S3D's classifier is dropout + a 1x1x1 convolution, then a
                # mean over time and space: exactly torchvision's forward.
                logits = torch.mean(m.classifier(pooled), dim=(2, 3, 4))[0]
                feat = torch.mean(pooled, dim=(2, 3, 4))[0]
            else:
                if rgb.shape[1:3] == INPUT_HW:
                    # Already 224x224, where the HF processor's resize and
                    # crop are identities: rescale + normalise on the GPU
                    # (bit-identical to the processor, and much faster).
                    x = torch.from_numpy(rgb).to(self.device).float().div_(255.0)
                    x = x.permute(0, 3, 1, 2).unsqueeze(0).contiguous()     # (1,T,C,H,W)
                    x = (x - self._vmae_mean) / self._vmae_std
                else:
                    x = self.processor(list(rgb), return_tensors="pt")["pixel_values"].to(self.device)
                m = self.model
                seq = m.videomae(pixel_values=x.to(m.dtype))[0]
                # As VideoMAEForVideoClassification.forward: mean-pooled
                # tokens through fc_norm, else the first token.
                feat = m.fc_norm(seq.mean(1)) if m.fc_norm is not None else seq[:, 0]
                logits = m.classifier(feat)[0]
                feat = feat[0]
        return logits.float(), feat.float().cpu().numpy()


def prepare_frame(frame_bgr: np.ndarray, bbox_xyxy=None, single_person: bool = True) -> np.ndarray | None:
    """One frame as the live classifier sees it, 224x224.

    One person in view: the whole frame (portrait frames cropped to a square
    around the person). Several: a crop around the person, padded by a
    quarter of the box on each side (as the pipeline's orchestrator does).
    """
    h, w = frame_bgr.shape[:2]
    if single_person or bbox_xyxy is None:
        src = frame_bgr
        if h > w * 1.2:                                     # _PORTRAIT_RATIO
            cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2 if bbox_xyxy is not None else h / 2
            y0 = int(min(max(0, cy - w / 2), h - w))
            src = frame_bgr[y0:y0 + w]
    else:
        x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
        px, py = int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)
        src = frame_bgr[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
    if src.size == 0:
        return None
    return cv2.resize(src, INPUT_HW, interpolation=cv2.INTER_LINEAR)
