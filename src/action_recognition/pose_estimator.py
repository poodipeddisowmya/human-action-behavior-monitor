"""Pose estimation with MediaPipe Pose (Lite).

Runs per person crop. 33 landmarks per person, ~5 ms on CPU. Output feeds the
skeleton-based ST-GCN classifier.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import mediapipe as mp
except ImportError:  # graceful fallback so tests can import without the dep
    mp = None


@dataclass
class Pose:
    keypoints: np.ndarray   # shape (33, 3) → x, y, visibility (normalized 0-1)


class PoseEstimator:
    def __init__(self, model_complexity: int = 0):
        if mp is None:
            raise ImportError("mediapipe is not installed. See requirements.txt")
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def estimate(self, crop_bgr: np.ndarray) -> Pose | None:
        rgb = crop_bgr[..., ::-1]
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return None
        kp = np.array(
            [[lm.x, lm.y, lm.visibility] for lm in result.pose_landmarks.landmark],
            dtype=np.float32,
        )
        return Pose(keypoints=kp)

    def close(self) -> None:
        self._pose.close()
