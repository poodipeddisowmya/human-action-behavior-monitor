"""Fall detection from the person's box and skeleton over time.

A fall is a change of state, not a speed: someone who was upright is, within
a second or so, lying on the floor and stays there. Measured on 9 real falls
(UR Fall Detection Dataset, front camera; one Mixkit clip) and 4 everyday
clips from the same rooms (crouching, picking something up, sitting on a
bed), 2026-11-18:

                                  falls (9)        everyday (4)
  box width / height at the end   1.26 - 2.40      0.56 - 0.86
  torso angle from vertical       57 - 85 deg (7)  5 - 33 deg
  upright -> lying-wide           0.13 - 0.60 s    never
  fastest drop of the box top     0.46 - 1.39      0.38 - 0.64  (body-heights/s)

Drop speed overlaps; the end state does not. So: the person must have been
seen upright (box taller than wide), then be lying (box at least 1.2x wider
than tall, and torso near horizontal or height collapsed to under 60% of
standing; shoulders and hips must be visible)
within FALL_WINDOW_S, and stay lying for CONFIRM_S. "Falling down" then
shows for as long as they stay down, and at least HOLD_S.

Not caught: a fall that ends sitting or kneeling against something (the box
stays tall), or one hidden behind furniture. A quick, deliberate lie-down on
a sofa does look like a fall here; for a safety alert that is the right side
to err on.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .action_classifier import ActionPrediction

LABEL = "falling down"

UPRIGHT_MAX_ASPECT = 0.9     # width / height below this: upright
LYING_MIN_ASPECT = 1.2       # at or above this: lying
LYING_MIN_TORSO_DEG = 45.0   # torso angle from vertical, when visible
COLLAPSED_HEIGHT = 0.6       # ...or box height at most this share of standing height
STANDING_MEMORY_S = 3.0      # standing height = tallest upright box in this window
FALL_WINDOW_S = 1.5          # upright -> lying within this (measured 0.13-0.6 s)
CONFIRM_S = 0.4              # lying this long first (0.3 and 0.4 both: 9/9 falls, 0 false)
LYING_SHARE = 0.7            # of frames (with a torso) during CONFIRM_S
HOLD_S = 5.0                 # shown at least this long once confirmed
GET_UP_S = 1.0               # upright this long ends it
_KP_MIN = 0.4
_FORGET_S = 60.0             # a track unseen this long is dropped


def torso_angle(keypoints) -> float | None:
    """Shoulder-mid to hip-mid angle from vertical, degrees; None if unseen."""
    if keypoints is None or len(keypoints) < 13:
        return None
    pts = [keypoints[i] for i in (5, 6, 11, 12)]
    if min(float(p[2]) for p in pts) < _KP_MIN:
        return None
    sx, sy = (pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2
    hx, hy = (pts[2][0] + pts[3][0]) / 2, (pts[2][1] + pts[3][1]) / 2
    return math.degrees(math.atan2(abs(hx - sx), abs(hy - sy) + 1e-6))


@dataclass
class _Track:
    seen_at: float = 0.0
    box: tuple = (0.0, 0.0, 0.0, 0.0)
    last_upright: float | None = None
    upright_heights: deque = field(default_factory=lambda: deque(maxlen=128))  # (t, box height)
    lying_since: float | None = None
    lying_frames: int = 0          # since lying_since: lying / all with a torso
    seen_frames: int = 0
    upright_since: float | None = None
    fallen_at: float | None = None
    samples: deque = field(default_factory=lambda: deque(maxlen=64))


class FallDetector:
    def __init__(self) -> None:
        self._tracks: dict[int, _Track] = {}

    def update(self, person_id: int, xyxy, keypoints, t: float) -> ActionPrediction | None:
        """Feed one observation (t in seconds, video or wall time); returns
        a 'falling down' prediction while a confirmed fall lasts."""
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        aspect = (x2 - x1) / max(1.0, y2 - y1)
        angle = torso_angle(keypoints)
        tr = self._tracks.get(person_id)
        if tr is None:
            tr = self._tracks[person_id] = self._inherit((x1, y1, x2, y2), t)
            for pid in [p for p, o in self._tracks.items() if t - o.seen_at > _FORGET_S]:
                del self._tracks[pid]
        tr.seen_at, tr.box = t, (x1, y1, x2, y2)

        # Both states need the torso itself in view: a box alone can be an
        # arm reaching into a close-up (demo-cooking-3: a hand at the pot,
        # box 1.2-1.5x wider than tall, read as a fall).
        upright = angle is not None and aspect < UPRIGHT_MAX_ASPECT and angle < LYING_MIN_TORSO_DEG
        # Lying: a wide box, and either a horizontal torso or a body that has
        # collapsed to well under its standing height. The second matters for
        # a fall towards or away from the camera, where the torso looks short
        # and steep (urfall-fall-03: torso 21-47 deg, height 190 -> 75 px).
        # Standing height: the tallest upright box of the last few seconds
        # (the last upright frame is often already on the way down).
        standing = max((h for ts, h in tr.upright_heights if t - ts <= STANDING_MEMORY_S), default=0.0)
        collapsed = standing > 0 and (y2 - y1) <= COLLAPSED_HEIGHT * standing
        lying = angle is not None and aspect >= LYING_MIN_ASPECT and (
            angle >= LYING_MIN_TORSO_DEG or collapsed)

        if upright:
            tr.last_upright = t
            tr.upright_heights.append((t, y2 - y1))
            tr.lying_since = None
            tr.upright_since = tr.upright_since if tr.upright_since is not None else t
        else:
            tr.upright_since = None
        if lying and tr.lying_since is None:
            tr.lying_since, tr.lying_frames, tr.seen_frames = t, 0, 0
        if tr.lying_since is not None and angle is not None:
            tr.seen_frames += 1
            tr.lying_frames += lying
        # A new fall: lying began soon after being upright, and has held in
        # most frames since. One "lying" frame is not enough: in a crowd the
        # skeleton jumps between people (hd-dancing-3: a dancer with arms out,
        # torso read as 0, 79, 40, 1 deg in consecutive frames).
        if (tr.fallen_at is None and tr.lying_since is not None and tr.last_upright is not None
                and tr.lying_since - tr.last_upright <= FALL_WINDOW_S
                and t - tr.lying_since >= CONFIRM_S):
            if tr.lying_frames >= LYING_SHARE * tr.seen_frames:
                tr.fallen_at = t
            else:
                tr.lying_since = None         # it did not hold: start over
        # Ongoing: until they have been up again for a while, and at least HOLD_S.
        if tr.fallen_at is not None:
            got_up = tr.upright_since is not None and t - tr.upright_since >= GET_UP_S
            if got_up and t - tr.fallen_at >= HOLD_S:
                tr.fallen_at = None
                return None
            return ActionPrediction(LABEL, 0.9)
        return None

    def _inherit(self, box: tuple, t: float) -> _Track:
        """A new track ID where one just vanished is the same person.

        A falling body changes shape so fast that the tracker often gives it
        a new ID mid-fall; starting that ID with no history ("never seen
        upright") would miss exactly the falls that matter.
        """
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        size = max(box[2] - box[0], box[3] - box[1], 1.0)
        best, best_d = None, None
        for pid, tr in self._tracks.items():
            if not (0 < t - tr.seen_at <= FALL_WINDOW_S):
                continue                  # still being tracked, or gone too long
            ox, oy = (tr.box[0] + tr.box[2]) / 2, (tr.box[1] + tr.box[3]) / 2
            osize = max(tr.box[2] - tr.box[0], tr.box[3] - tr.box[1], 1.0)
            d = math.hypot(cx - ox, cy - oy)
            if d <= max(size, osize) and (best_d is None or d < best_d):
                best, best_d = pid, d
        if best is None:
            return _Track()
        return self._tracks.pop(best)

    def forget(self, person_id: int) -> None:
        self._tracks.pop(person_id, None)

    def reset(self) -> None:
        self._tracks.clear()
