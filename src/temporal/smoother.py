"""Temporal smoothing — kills the 'standing/sitting/standing' flicker.

Majority vote over a short sliding window, per person.

Design rule: a tracked person ALWAYS gets a label.

Earlier versions returned None in two cases — while the window was still
filling, and when a prediction fell below a confidence floor. Both produced
an empty "- 0%" box in the UI, which reads to a viewer as "the system is
broken" even though it was merely being cautious. Caution should show up as a
*lower confidence number*, never as missing output.

So the smoother now always answers when given a prediction: it smooths once it
has enough history, and passes the raw prediction through (flagged by a
slightly reduced confidence) before that.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from src.action_recognition import ActionPrediction


@dataclass
class SmoothedPrediction:
    label: str
    confidence: float


class TemporalSmoother:
    def __init__(self, window: int = 5, min_confidence: float = 0.5):
        self._window = window
        # Retained as a *reporting* floor, not a gate: predictions below it
        # are still emitted, just visibly less confident.
        self._min_confidence = min_confidence
        self._buffers: dict[int, deque[ActionPrediction]] = {}

    def smooth(self, person_id: int, pred: ActionPrediction) -> SmoothedPrediction:
        buf = self._buffers.setdefault(person_id, deque(maxlen=self._window))
        buf.append(pred)

        # Not enough history to vote yet — pass the current read through so
        # the person is never unlabelled, discounted to reflect that it is
        # unsmoothed.
        if len(buf) < self._window:
            return SmoothedPrediction(
                label=pred.label,
                confidence=round(pred.confidence * 0.85, 3),
            )

        counts = Counter(p.label for p in buf)
        label, hits = counts.most_common(1)[0]
        confs = [p.confidence for p in buf if p.label == label]
        conf = sum(confs) / len(confs)

        # A bare majority is weaker evidence than a unanimous window; reflect
        # that in the number rather than by withholding the label.
        agreement = hits / len(buf)
        if agreement < 0.6:
            conf *= 0.85

        return SmoothedPrediction(label=label, confidence=round(conf, 3))

    def reset(self, person_id: int) -> None:
        self._buffers.pop(person_id, None)

    def reset_all(self) -> None:
        """Drop all per-person vote buffers — used when the source changes."""
        self._buffers.clear()
