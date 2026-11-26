"""What the running monitor can recognise: built-in, described in words, and
trained on your own clips (docs/TRAINING.md), with the trained ones'
measured accuracy."""
from __future__ import annotations

from fastapi import APIRouter

from config import settings
from src.action_recognition.target_vocabulary import TARGET_CATEGORIES
from src.api.runtime import runtime

router = APIRouter()


@router.get("")
async def activities() -> dict:
    p = runtime.pipeline
    clf = getattr(p, "_classifier", None)
    ov = getattr(p, "_open_vocab", None)
    custom = getattr(clf, "_custom", None)
    trained = None
    if custom is not None:
        meta = custom.head.meta
        v = meta.get("validation", {})
        trained = {
            "backend": custom.head.backend,
            "activities": custom.activities,
            "phrases": meta.get("phrases", {}),
            "trained_at": meta.get("trained_at"),
            "clips": meta.get("clips", {}),
            "threshold": custom.threshold,
            # Measured on clips held out from training (cross-validation).
            "held_out": {"activities": v.get("activities"),
                         "false_alarm_clips": v.get("false_alarm_clips"),
                         "other_clips": v.get("other_clips")},
        }
    return {
        "backend": getattr(clf, "_backend", settings.model.deep_backend),
        "built_in": sorted(TARGET_CATEGORIES),
        "described_in_words": sorted(getattr(ov, "activities", {}) or {}) if getattr(ov, "_ready", False) else [],
        "trained": trained,
    }
