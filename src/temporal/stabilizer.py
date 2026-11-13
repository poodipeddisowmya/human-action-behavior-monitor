"""Display stability: stop posture labels flickering on screen.

The smoother votes over the last few detection updates. On a GPU those
arrive 25-60 times a second, so its window covers ~0.1 s and a pose rule
wavering between "walking" and "running" on a low-resolution clip still
flipped the on-screen label several times a second (measured on a T4:
14.6 label changes per 10 s across the demo clips).

This sits after the smoother and decides only what is SHOWN:
  * an activity (cooking, writing, ...) or "falling" replaces the display
    at once — activities have already passed the deep model's own voting,
    and a fall must never wait;
  * otherwise the display shows the label that dominates the last
    `window_s` seconds: it switches when another label holds at least half
    of that window, or when the shown one has all but disappeared from it.

A first version required a new posture to persist unbroken for the whole
window. When the raw label alternated (walking / running / sitting on a
320x240 clip) no candidate was ever unbroken, so the display stuck on
whatever came first: a walking clip showed "running" 81% of the time.
Time is the video clock, so the behaviour is the same at any frame rate.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from .smoother import SmoothedPrediction

POSTURES = frozenset({"standing", "sitting", "walking", "running", "lying",
                      "kneeling", "bending", "crawling"})

_SWITCH_SHARE = 0.5     # a challenger needs half the window...
_GONE_SHARE = 0.2       # ...or the shown label must have nearly vanished


@dataclass
class _State:
    shown: SmoothedPrediction
    history: deque = field(default_factory=deque)     # (t, label)


class LabelStabilizer:
    def __init__(self, window_s: float = 0.8):
        self._window = max(0.0, float(window_s))
        self._state: dict[int, _State] = {}

    def update(self, person_id: int, pred: SmoothedPrediction, t: float) -> SmoothedPrediction:
        st = self._state.get(person_id)
        if st is None or self._window == 0:
            st = self._state[person_id] = _State(pred)
            st.history.append((t, pred.label))
            return pred

        st.history.append((t, pred.label))
        while st.history and st.history[0][0] < t - self._window:
            st.history.popleft()

        if pred.label == st.shown.label:
            st.shown = pred                               # keep confidence fresh
            return pred
        if pred.label not in POSTURES or st.shown.label.startswith("detecting"):
            st.shown = pred                               # activity / fall / first read
            return pred

        counts = Counter(label for _, label in st.history)
        n = len(st.history)
        top, hits = counts.most_common(1)[0]
        if top != st.shown.label and (hits / n >= _SWITCH_SHARE
                                      or counts[st.shown.label] / n <= _GONE_SHARE):
            st.shown = SmoothedPrediction(top, pred.confidence if top == pred.label
                                          else st.shown.confidence)
        return st.shown

    def reset(self, person_id: int) -> None:
        self._state.pop(person_id, None)

    def reset_all(self) -> None:
        self._state.clear()
