"""Fast pose-driven action classifier — the production hot path.

Reads the 17 COCO keypoints that YOLOv8-pose already produced (zero extra cost)
and infers a rich set of common daily activities from geometry alone:

    posture     : standing · sitting · lying · kneeling · bending
    motion      : walking · running · falling
    hand-based  : drinking · using phone · waving · clapping · reading
    body-based  : push-up · crawling

Cost per person per frame: <1 ms of pure Python. Always fast, always
deterministic, no model download.

For context-heavy activities (cooking, playing music, dancing, ...) the pose
alone can't decide. In that case this classifier returns "unknown" and the
composed VideoMAEClassifier hands off to the deep model.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import settings


@dataclass
class ActionPrediction:
    label: str
    confidence: float
    # True when the skeleton was unreadable and this is box motion alone.
    degraded: bool = False


# --- COCO-17 keypoint indices (YOLOv8-pose output) ---
NOSE           = 0
L_EYE, R_EYE   = 1, 2
L_EAR, R_EAR   = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW       = 7, 8
L_WRIST, R_WRIST       = 9, 10
L_HIP, R_HIP           = 11, 12
L_KNEE, R_KNEE         = 13, 14
L_ANKLE, R_ANKLE       = 15, 16

# --- Motion thresholds (body-heights per second) ---
# 0.20 was set by eye and sat ABOVE the measured speed of a person walking
# across a room in portrait phone video (0.164), so real walking was reported
# as "standing". Measured medians (scripts/measure_walk_speed.py):
#
#     typing at a desk    0.032      walking across room   0.164
#     seated, drinking    0.041      lowering into chair   0.151
#
# The stationary and moving groups separate cleanly (~4x). 0.09 sits in the
# empty band between them. Note what this threshold canNOT do: walking and
# sitting-down are indistinguishable here, and stay that way under direction
# decomposition and longer measurement baselines (both tested, both failed —
# at a 1s baseline they invert). Telling those two apart needs limb kinematics,
# which is the deep model's job, not this one's.
# These MUST be calibrated at the same window the classifier measures over
# (action_window frames, buf[0] to buf[-1]), not frame-to-frame. Speed falls
# as the window lengthens — running measures 0.536 frame-to-frame but 0.368
# over 16 frames — so thresholds picked from per-frame numbers sit too high
# and everything collapses one category downward.
#
# Medians at the 16-frame window (scripts/measure_walk_speed.py --lags 16):
#     sitting              0.059
#     mopping (in place)   0.034
#     standing at counter  0.110
#     walking              0.160, 0.178   (two different clips)
#     running              0.368
#
# 0.12 and 0.27 sit in the gaps. The earlier values (0.20/1.25) were set by
# eye: 1.25 was unreachable, so "running" never fired at all, and 0.20 sat
# above walking, so walkers were reported as standing.
#
# Note how tight standing-at-a-counter (0.110) is against walking (0.160):
# someone working at a worktop genuinely does translate a little. This
# boundary is the least comfortable of the three and is the one to revisit
# first if standing/walking confusion shows up in the field.
_STAND_MAX = 0.12
_WALK_MAX  = 0.27
_FALL_VDROP = 1.4

# The thresholds above were calibrated over 16 CONSECUTIVE frames of ~30 fps
# video, i.e. about half a second. The history buffer is 16 SAMPLES, though,
# and the live detection thread only samples ~7-8 frames a second — so live,
# the same buffer spanned ~2 s, speed was averaged over four times the
# calibrated baseline ("speed falls as the window lengthens"), and the live
# server answered differently from the offline harness on the same clip
# (measured, 11.58.18 phone clip of a man walking: 53% walking at full frame
# rate, 6% at live sampling). Measuring over a fixed span of VIDEO TIME makes
# the number mean the same thing at any detection rate. At 30 fps and full
# rate this selects the same ~16 frames as before.
_MOTION_WINDOW_S = 0.55

# Walking straight toward (or away from) the camera barely moves the box
# centre — the box GROWS instead. Centre displacement alone called that
# "standing". Measured growth rate, ln(size ratio) per second over the window
# above (scratch measurements, 8 samples/s):
#
#     walking toward camera (phone)   0.40 - 0.58
#     walking toward camera (film)    0.22 - 0.32
#     seated / smoking / typing       0.00 - 0.06
#
# Height and width must BOTH change, in the same direction: a squat or
# sitting down shrinks height only, a raised arm widens only, and neither is
# the person coming closer.
_APPROACH_MIN = 0.25

# --- Keypoint visibility floor ---
# 0.35 is too strict for far-away or fast-moving people; 0.2 keeps most
# real keypoints while still filtering pure noise.
_MIN_VIS = 0.20
_MIN_VIS_HAND = 0.15    # even looser for wrists/elbows (often occluded)

# Start classifying almost immediately. A newly-detected person showing a
# blank "- 0%" label reads as "the system is broken", even when it is merely
# waiting. Two frames is enough for a motion estimate, and an honest
# low-confidence "standing" beats an empty box.
_MIN_HISTORY = 2
_UNKNOWN = "unknown"    # signal to composer that deep model should try

# --- Skeleton plausibility ---
# YOLOv8-pose will happily emit a skeleton for a crop that contains no torso
# at all. Measured on a kitchen close-up of two hands: shoulders came back at
# 0.01 / 0.00 visibility while "hips" were reported at 0.67 / 0.49 — entirely
# hallucinated. Trusting that produced a confident "standing 87%" on a frame
# of someone chopping an onion.
#
# So before ANY geometric reasoning we require the skeleton to be coherent:
# both shoulders AND both hips actually visible, plus a minimum number of
# landmarks overall. If it fails, we return None ("unknown") and let the deep
# model — which reads pixels, not joints — answer instead.
_MIN_TORSO_VIS = 0.25      # each of L/R shoulder and L/R hip must clear this

# Floor-posture rules (push-up, crawling) need the LOWER body genuinely in
# view, not merely "plausible".
#
# On a waist-up shot YOLOv8-pose still reports hips, at 0.37-0.49 visibility,
# placed near the bottom edge of the frame — close enough to shoulder height
# that the torso measured "horizontal" and the person was called push-up at
# 0.82. Measured (visibility, 10th-90th percentile):
#
#                              hips         best knee
#     real push-ups            0.95-0.99    0.88-0.98
#     writing at a board       0.41-0.47    0.07
#     chopping at a counter    0.38-0.49    0.07-0.12
#     seated, drinking         0.37-0.49    0.06-0.07
#
# That misfire put "exercising" on screen for 8% of demo-writing and 21-40%
# of a drinking clip. The gap is wide, so the bars sit well inside it.
_FLOOR_MIN_HIP_VIS = 0.70
_FLOOR_MIN_KNEE_VIS = 0.50
_MIN_VISIBLE_KEYPOINTS = 7  # of 17, to reject partial-body crops


@dataclass
class _Sample:
    keypoints: Optional[np.ndarray]              # shape (17, 3)
    bbox: tuple[float, float, float, float]      # x1, y1, x2, y2 in frame coords
    t: float


class SkeletonHeuristicClassifier:
    """Rich rule-based classifier over YOLOv8-pose keypoints."""

    def __init__(self, window: int | None = None):
        self._window = window or settings.model.action_window
        self._history: dict[int, deque[_Sample]] = {}

    # ------------------------------------------------------------------
    def update(
        self,
        person_id: int,
        keypoints: np.ndarray | None,
        bbox_xyxy: tuple[float, float, float, float],
        crop_bgr=None,   # unused — accepted so signature matches VideoMAE backend
        timestamp: float | None = None,
        single_person: bool = False,   # unused — see VideoMAEClassifier.update
        frame_hw=None,                 # unused — see VideoMAEClassifier.update
    ) -> ActionPrediction | None:
        # `timestamp` must be the VIDEO presentation time when the source is a
        # file. Using wall-clock there makes speed depend on how fast we happen
        # to be processing: the detection thread runs slower than playback, so
        # dt comes out too large, every speed is under-reported, and walking
        # gets labelled "standing" — worse on a loaded machine, better on an
        # idle one. Same clip, different answer, which is not acceptable.
        # For a live camera there is no separate video clock, so wall-clock is
        # the right answer and remains the default.
        buf = self._history.setdefault(person_id, deque(maxlen=self._window))
        t = time.time() if timestamp is None else timestamp
        buf.append(_Sample(keypoints=keypoints, bbox=tuple(bbox_xyxy), t=t))
        if len(buf) < _MIN_HISTORY:
            return None
        return self._classify(buf)

    def reset(self, person_id: int) -> None:
        self._history.pop(person_id, None)

    def reset_all(self) -> None:
        """Drop all per-person history — used when the video source changes."""
        self._history.clear()

    # ------------------------------------------------------------------
    def _classify(self, buf: deque[_Sample]) -> ActionPrediction | None:
        latest = buf[-1]
        # Oldest sample inside the calibrated time span (see _MOTION_WINDOW_S).
        first = buf[0]
        for s in buf:
            if latest.t - s.t <= _MOTION_WINDOW_S:
                first = s
                break
        if first is latest and len(buf) >= 2:
            first = buf[-2]       # samples sparser than the window: use the gap
        dt = latest.t - first.t
        if dt < 0.1:
            return None

        # ----- bbox motion in body-heights / second -----
        cx_l, cy_l, h_l = _bbox_stats(latest.bbox)
        cx_f, cy_f, h_f = _bbox_stats(first.bbox)
        body_h = max(h_l, h_f, 1.0)
        dx_bh = (cx_l - cx_f) / body_h
        dy_bh = (cy_l - cy_f) / body_h
        speed = math.hypot(dx_bh, dy_bh) / dt
        v_speed = dy_bh / dt
        approach = _approach_rate(first.bbox, latest.bbox, dt)

        # ----- 1. Safety: falling (fast downward motion) -----
        if v_speed > _FALL_VDROP:
            conf = _score(v_speed - _FALL_VDROP, spread=1.0, base=0.65)
            return ActionPrediction("falling", conf)

        # ----- posture / hand geometry from the pose -----
        kp = latest.keypoints

        # No skeleton, or an implausible one (partial-body crop, or YOLO
        # hallucinating joints on a close-up). We must NOT reason about
        # posture from garbage geometry — but bounding-box motion is measured
        # from the track, not the joints, and remains perfectly valid.
        #
        # Returning None here was a real production bug: a person with an
        # unreadable skeleton got no label at all, so the UI showed an empty
        # box. A tracked person should ALWAYS carry a label; the honest answer
        # when joints are unreadable is their motion state, at reduced
        # confidence to reflect that posture could not be verified.
        if kp is None or not _skeleton_is_plausible(kp):
            # Running is only claimed with an upright skeleton behind it: a
            # close-up at a desk (no legs in view) moves its box fast with
            # every lean and turn.
            return self._motion_label(speed, degraded=True, approach=approach,
                                      allow_run=False)

        # Check crawling / push-up BEFORE lying — both have a horizontal
        # torso, but their intent is different (crawling has hands and knees
        # on the floor and body moving; lying is static and full length).
        crawl = self._is_crawling(kp, body_h)
        if crawl:
            return crawl

        pushup = self._is_pushup(kp, body_h)
        if pushup:
            return pushup

        posture = self._read_posture(kp, body_h)
        if posture and posture[0] == "lying":
            return ActionPrediction(*posture)

        # ----- kneeling / bending (posture-based) -----
        if posture and posture[0] in ("sitting", "kneeling", "bending"):
            if speed > _WALK_MAX:
                # Moving fast while the skeleton reads seated or bent: most
                # likely a posture misread of someone walking, or someone
                # swivelling in an office chair. Never "running": in
                # the office clips (Pexels RDNE set, 2026-11-23) that was a
                # seated typist read as running 13% of the time.
                return self._motion_label(speed, allow_run=False)
            return ActionPrediction(*posture)

        # NOTE: hand-based rules (drinking / using phone / waving / clapping /
        # reading) used to run here and have been REMOVED.
        #
        # They were measured against real footage and were badly wrong:
        #   typing at a laptop  -> "clapping" on 34 of 36 frames
        #   sitting down        -> "waving"   on 32 of 36 frames
        #   walking             -> "clapping" on  9 of 36 frames
        #   actually drinking   -> never fired at all
        #
        # The reason is fundamental, not a tuning problem: in 17-keypoint space
        # typing, clapping, reading and writing are the same pose — two hands
        # together in front of the chest. Wrist geometry cannot separate them,
        # and no threshold makes it possible.
        #
        # The deep model CAN separate them, because it sees the laptop: it
        # labels that same clip "using computer" at 79-95%. So these activities
        # now belong exclusively to the deep model, and the pose engine sticks
        # to what body geometry genuinely determines: posture and motion.

        # ----- motion labels -----
        return self._motion_label(speed, approach=approach)

    # ------------------------------------------------------------------
    # Posture from pose
    # ------------------------------------------------------------------
    def _read_posture(self, kp: np.ndarray, body_h: float) -> tuple[str, float] | None:
        hx, hy, hv = _mid(kp, L_HIP, R_HIP)
        sx, sy, sv = _mid(kp, L_SHOULDER, R_SHOULDER)
        ax, ay, av = _mid(kp, L_ANKLE, R_ANKLE)
        kx, ky, kv = _mid(kp, L_KNEE, R_KNEE)
        if hv < _MIN_VIS or sv < _MIN_VIS:
            return None
        torso = abs(hy - sy) / body_h
        legs  = abs(ay - hy) / body_h if av >= _MIN_VIS else 0.0

        # Lying: hips and shoulders at similar height (body horizontal)
        if av >= _MIN_VIS and torso < 0.15 and legs < 0.15:
            return ("lying", 0.85)

        # Sitting: knees level with or above the hips.
        #
        # This used to also require `av >= _MIN_VIS` — ankles visible — which
        # made it dead code in practice. A seated person's ankles are the most
        # occluded joint there is (under a desk, behind a chair, out of frame):
        # measured ankle visibility on demo-basic-sit is 0.00, so the rule
        # never fired once and every seated person came back "standing".
        #
        # Knees are visible when ankles are not, and separate cleanly
        # (scripts/diag_sitting.py, median (knee_y - hip_y)/body_h):
        #     seated, side view          -0.707
        #     seated at desk, drinking   -0.106
        #     standing / walking         +0.238
        # Image y grows downward, so negative = knees above hips = seated.
        knee_above_hip = (ky - hy) / body_h < 0.05
        if kv >= _MIN_VIS and knee_above_hip and torso > 0.18:
            return ("sitting", 0.75)
        # Legs visibly foreshortened, when we do happen to have ankles.
        if kv >= _MIN_VIS and av >= _MIN_VIS and legs < 0.28 and torso > 0.18:
            return ("sitting", 0.75)

        # Kneeling: knees on the ground level, ankles behind
        if kv >= _MIN_VIS and av >= _MIN_VIS and abs(ky - ay) / body_h < 0.06:
            return ("kneeling", 0.7)

        # Bending: nose below hip level while otherwise upright
        if kp[NOSE, 2] >= _MIN_VIS and kp[NOSE, 1] > hy - 0.05 * body_h:
            return ("bending", 0.7)

        return None

    # ------------------------------------------------------------------
    # Push-up detection
    # ------------------------------------------------------------------
    def _is_pushup(self, kp: np.ndarray, body_h: float) -> ActionPrediction | None:
        hx, hy, hv = _mid(kp, L_HIP, R_HIP)
        sx, sy, sv = _mid(kp, L_SHOULDER, R_SHOULDER)
        wx_l, wy_l, wv_l = kp[L_WRIST]
        wx_r, wy_r, wv_r = kp[R_WRIST]
        ax, ay, av = _mid(kp, L_ANKLE, R_ANKLE)
        if min(hv, sv) < _MIN_VIS:
            return None
        if not _lower_body_in_view(kp):
            return None
        # Body horizontal
        torso_flat = abs(hy - sy) / body_h < 0.20
        if not torso_flat:
            return None
        # Wrists visible and below the shoulders (arms extended to floor)
        if wv_l < _MIN_VIS and wv_r < _MIN_VIS:
            return None
        wrist_y = max(wy_l if wv_l >= _MIN_VIS else -1e9,
                      wy_r if wv_r >= _MIN_VIS else -1e9)
        if wrist_y < sy:
            return None
        # Ankles also visible + roughly level with hips (feet on floor)
        if av >= _MIN_VIS and abs(ay - hy) / body_h > 0.30:
            return None
        return ActionPrediction("push-up", 0.82)

    # ------------------------------------------------------------------
    def _is_crawling(self, kp: np.ndarray, body_h: float) -> ActionPrediction | None:
        hx, hy, hv = _mid(kp, L_HIP, R_HIP)
        sx, sy, sv = _mid(kp, L_SHOULDER, R_SHOULDER)
        kx, ky, kv = _mid(kp, L_KNEE, R_KNEE)
        wx_l, wy_l, wv_l = kp[L_WRIST]
        wx_r, wy_r, wv_r = kp[R_WRIST]
        if min(hv, sv, kv) < _MIN_VIS:
            return None
        if not _lower_body_in_view(kp):
            return None
        # Torso must be genuinely HORIZONTAL: shoulders and hips at nearly the
        # same height.
        #
        # This threshold was 0.32, which was a serious false-positive bug: an
        # ordinary upright adult measures 0.28-0.32 on this ratio, so "crawling"
        # fired on people who were simply standing or walking. Measured on real
        # footage — walking person 0.29, person sitting down 0.30, baby actually
        # crawling ~0.02. 0.15 separates them cleanly.
        torso_flat = abs(hy - sy) / body_h < 0.15

        # Hands are down at roughly hip height or lower — i.e. bearing weight.
        # (A knee-below-hip test is useless here: that is true when standing.)
        wrist_down = ((wv_l >= _MIN_VIS_HAND and wy_l > hy - 0.05 * body_h) or
                      (wv_r >= _MIN_VIS_HAND and wy_r > hy - 0.05 * body_h))

        # Knees roughly level with hips, not far below — the signature of being
        # down on all fours rather than upright.
        knees_level = abs(ky - hy) / body_h < 0.20

        if torso_flat and wrist_down and knees_level:
            return ActionPrediction("crawling", 0.78)
        return None

    # ------------------------------------------------------------------
    # Hand-based activities (drinking, phone, waving, clapping, reading)
    # ------------------------------------------------------------------
    def _hand_activities(self, kp: np.ndarray, body_h: float):
        nose_x, nose_y, nose_v = kp[NOSE]
        sx, sy, sv = _mid(kp, L_SHOULDER, R_SHOULDER)
        hx, hy, hv = _mid(kp, L_HIP, R_HIP)
        if sv < _MIN_VIS:
            return

        # helper: is wrist near a target point (units of body-h)
        def near(wv, wx, wy, tx, ty, thr):
            if wv < _MIN_VIS_HAND: return False
            return math.hypot((wx - tx), (wy - ty)) / body_h < thr

        # helper: wrist above head?
        def raised(wv, wy):
            return wv >= _MIN_VIS_HAND and wy < sy - 0.15 * body_h

        wx_l, wy_l, wv_l = kp[L_WRIST]
        wx_r, wy_r, wv_r = kp[R_WRIST]
        ex_l, ey_l, ev_l = kp[L_ELBOW]
        ex_r, ey_r, ev_r = kp[R_ELBOW]

        # 1. Drinking: a hand raised to the mouth.
        #
        # Straight-line wrist-to-nose distance does NOT work here — measured on
        # real footage, a person drinking reads 0.26-0.34 while a person
        # walking with arms swinging reads 0.33-0.42. The ranges overlap, so
        # any single distance threshold either misses drinking or fires on
        # walking.
        #
        # VERTICAL position separates them cleanly: a hand raised to drink sits
        # at roughly face height, whereas a relaxed or swinging arm sits down
        # near the hips, well below the nose. So require the wrist to be near
        # the face in height AND not far off in x.
        if nose_v >= _MIN_VIS:
            for wv, wx, wy in ((wv_l, wx_l, wy_l), (wv_r, wx_r, wy_r)):
                if wv < _MIN_VIS_HAND:
                    continue
                dy = abs(wy - nose_y) / body_h        # height agreement w/ face
                dx = abs(wx - nose_x) / body_h        # lateral offset from face
                if dy < 0.18 and dx < 0.25:
                    yield ("drinking", 0.75)
                    return

        # 2. Using phone: wrist near face, hand held up at eye/nose level, elbow bent
        if nose_v >= _MIN_VIS:
            for wv, wx, wy in ((wv_l, wx_l, wy_l), (wv_r, wx_r, wy_r)):
                if (wv >= _MIN_VIS_HAND and
                    abs(wx - nose_x) / body_h < 0.35 and
                    -0.10 < (nose_y - wy) / body_h < 0.30):
                    yield ("using phone", 0.68)
                    return

        # 3. Waving: one wrist high above shoulders
        if raised(wv_l, wy_l) or raised(wv_r, wy_r):
            yield ("waving", 0.7)
            return

        # 4. Clapping: both wrists in front, close together, near chest height
        if wv_l >= _MIN_VIS_HAND and wv_r >= _MIN_VIS_HAND:
            wrist_dist = math.hypot(wx_l - wx_r, wy_l - wy_r) / body_h
            wrist_mid_y = (wy_l + wy_r) / 2
            near_chest = sy - 0.05 * body_h < wrist_mid_y < hy
            if wrist_dist < 0.22 and near_chest:
                yield ("clapping", 0.7)
                return

        # 5. Reading: both hands in front at chest level, head tilted down
        if (wv_l >= _MIN_VIS_HAND and wv_r >= _MIN_VIS_HAND and nose_v >= _MIN_VIS):
            hands_mid_y = (wy_l + wy_r) / 2
            hands_below_shoulders = hands_mid_y > sy + 0.05 * body_h
            hands_above_hips = hands_mid_y < hy
            head_tilted_down = nose_y > (sy - 0.03 * body_h)
            hands_close = math.hypot(wx_l - wx_r, wy_l - wy_r) / body_h < 0.40
            if hands_below_shoulders and hands_above_hips and head_tilted_down and hands_close:
                yield ("reading", 0.65)
                return

    # ------------------------------------------------------------------
    def _motion_label(self, speed: float, degraded: bool = False,
                      approach: float = 0.0, allow_run: bool = True) -> ActionPrediction:
        """Motion state from bounding-box trajectory alone.

        `degraded` means the skeleton was unreadable, so this is a motion
        estimate with no posture verification behind it. We still answer —
        an empty label is never the right output for a tracked person — but
        we scale the confidence down to say so honestly.
        """
        if speed < _STAND_MAX and approach >= _APPROACH_MIN:
            # Not translating across the image, but steadily changing size:
            # walking along the camera axis (see _APPROACH_MIN).
            conf = _score(approach - _APPROACH_MIN, spread=0.30, base=0.65)
            label = "walking"
        elif speed < _STAND_MAX:
            # spread tracks the width of the band, so a person at rest still
            # scores near the top of it. (It was a fixed 0.2 while the band
            # was 0.20 wide; narrowing the band without this would have
            # quietly deflated confidence on every genuinely still person.)
            conf = _score(_STAND_MAX - speed, spread=_STAND_MAX, base=0.60)
            label = "standing"
        elif speed < _WALK_MAX or not allow_run:
            speed = min(speed, _WALK_MAX)
            mid = (_STAND_MAX + _WALK_MAX) / 2
            proximity = 1.0 - abs(speed - mid) / (_WALK_MAX - _STAND_MAX) * 2
            conf = _score(proximity, spread=1.0, base=0.65)
            label = "walking"
        else:
            conf = _score(speed - _WALK_MAX, spread=1.0, base=0.7)
            label = "running"

        if degraded:
            conf *= 0.7   # unverified posture -> visibly lower confidence
        return ActionPrediction(label, round(conf, 3), degraded=degraded)


def _skeleton_is_plausible(kp: np.ndarray) -> bool:
    """Is this skeleton coherent enough to reason about geometrically?

    Guards against two real failure modes seen in testing:
      * partial-body crops (hands only, head only) where most joints are absent
      * YOLOv8-pose hallucinating high-confidence joints that aren't in frame

    Requires BOTH shoulders and BOTH hips — a torso needs four corners; one
    real hip plus one imagined shoulder is not a torso.
    """
    if kp is None or kp.shape[0] < 17:
        return False
    torso = (kp[L_SHOULDER, 2], kp[R_SHOULDER, 2],
             kp[L_HIP, 2], kp[R_HIP, 2])
    if min(torso) < _MIN_TORSO_VIS:
        return False
    if int((kp[:, 2] >= _MIN_VIS).sum()) < _MIN_VISIBLE_KEYPOINTS:
        return False
    return True


def _lower_body_in_view(kp: np.ndarray) -> bool:
    """Both hips and at least one knee clearly visible — see _FLOOR_MIN_HIP_VIS."""
    return (min(kp[L_HIP, 2], kp[R_HIP, 2]) >= _FLOOR_MIN_HIP_VIS
            and max(kp[L_KNEE, 2], kp[R_KNEE, 2]) >= _FLOOR_MIN_KNEE_VIS)


def _approach_rate(first, latest, dt: float) -> float:
    """Rate at which the box is uniformly growing or shrinking, per second.

    Zero unless height and width change in the same direction — see
    _APPROACH_MIN for why both are required.
    """
    fw, fh = max(1.0, first[2] - first[0]), max(1.0, first[3] - first[1])
    lw, lh = max(1.0, latest[2] - latest[0]), max(1.0, latest[3] - latest[1])
    gh, gw = math.log(lh / fh) / dt, math.log(lw / fw) / dt
    if gh * gw <= 0:
        return 0.0
    return min(abs(gh), abs(gw))


def _bbox_stats(b):
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2, (y1 + y2) / 2, max(1.0, y2 - y1)


def _mid(kps: np.ndarray, a: int, b: int) -> tuple[float, float, float]:
    return (
        (kps[a, 0] + kps[b, 0]) / 2,
        (kps[a, 1] + kps[b, 1]) / 2,
        (kps[a, 2] + kps[b, 2]) / 2,
    )


def _score(evidence: float, *, spread: float, base: float) -> float:
    if evidence <= 0:
        return base
    frac = min(1.0, evidence / spread)
    return round(base + (0.95 - base) * frac, 3)


# --------------------------------------------------------------------------
# Optional ST-GCN backend (kept for compatibility; auto-falls back)
# --------------------------------------------------------------------------

class STGCNClassifier:
    def __init__(self):
        self._weights_path = settings.models_dir / "st_gcn_kinetics.pt"
        self._available = self._weights_path.exists()
        self._fallback = SkeletonHeuristicClassifier()
        if self._available:
            import torch
            self._model = torch.jit.load(str(self._weights_path), map_location=settings.device)
            self._model.eval()
            self._torch = torch
            self._labels = settings.model.action_labels
            self._window = settings.model.action_window
            self._history: dict[int, deque[np.ndarray]] = {}

    def update(self, person_id, keypoints, bbox_xyxy, crop_bgr=None, timestamp=None,
               single_person=False, frame_hw=None):
        if not self._available or keypoints is None:
            return self._fallback.update(person_id, keypoints, bbox_xyxy,
                                         crop_bgr, timestamp)
        buf = self._history.setdefault(person_id, deque(maxlen=self._window))
        buf.append(keypoints)
        if len(buf) < self._window:
            return self._fallback.update(person_id, keypoints, bbox_xyxy,
                                         crop_bgr, timestamp)
        x = self._torch.from_numpy(np.stack(buf)).float().unsqueeze(0)
        x = x.permute(0, 3, 1, 2).unsqueeze(-1).to(settings.device)
        with self._torch.no_grad():
            logits = self._model(x)
            probs = self._torch.softmax(logits, dim=-1)[0]
        idx = int(probs.argmax())
        return ActionPrediction(self._labels[idx], float(probs[idx]))


ActionClassifier = STGCNClassifier
