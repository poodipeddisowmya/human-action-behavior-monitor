"""Person detector backed by pretrained YOLOv8 (Ultralytics)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# config first: it sets MPLBACKEND, which ultralytics' matplotlib import needs.
from config import settings
from ultralytics import YOLO  # noqa: E402

PERSON_CLASS_ID = 0  # COCO


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float


class PersonDetector:
    def __init__(self, weights: str | None = None, device: str | None = None):
        self._model = YOLO(weights or settings.model.detector_weights)
        self._device = device or settings.device
        self._conf = settings.model.detector_conf
        self._iou = settings.model.detector_iou

    def detect(self, image: np.ndarray) -> list[Detection]:
        result = self._model.predict(
            image,
            classes=[PERSON_CLASS_ID],
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=False,
        )[0]

        out: list[Detection] = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append(Detection(xyxy=(x1, y1, x2, y2), confidence=float(box.conf[0])))
        return out
