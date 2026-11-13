from .pose_estimator import PoseEstimator, Pose
from .action_classifier import (
    ActionPrediction,
    SkeletonHeuristicClassifier,
    STGCNClassifier,
)
from .videomae_classifier import VideoMAEClassifier

from config import settings


def _build_classifier():
    """Pick the classifier backend from `model_config.yaml`.

    Order of precedence:
        videomae         -> VideoMAEClassifier (400-class Kinetics)
        st_gcn_kinetics  -> STGCNClassifier (auto-fallback to heuristic)
        anything else    -> SkeletonHeuristicClassifier
    """
    choice = (settings.model.action_classifier or "").strip().lower()
    if choice == "videomae":
        return VideoMAEClassifier()
    if choice.startswith("st_gcn") or choice == "stgcn":
        return STGCNClassifier()
    return SkeletonHeuristicClassifier()


ActionClassifier = _build_classifier

__all__ = [
    "PoseEstimator",
    "Pose",
    "ActionClassifier",
    "ActionPrediction",
    "SkeletonHeuristicClassifier",
    "STGCNClassifier",
    "VideoMAEClassifier",
]
