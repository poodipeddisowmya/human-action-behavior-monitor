"""VideoMAE action classifier with restricted-vocabulary inference.

Design
------
Rather than taking the arg-max over all 400 Kinetics-400 classes, inference is
masked down to only the classes that map into our target categories
(see `target_vocabulary.py`). Two signals come out of one forward pass:

    coverage   how much of the model's total probability mass lands inside our
               vocabulary at all — i.e. "is this scene even one of the things
               we track?"
    confidence the renormalized probability of the winning category *within*
               our vocabulary — i.e. "given it's one of ours, which one?"

Both must clear their bar before a label is surfaced. If either fails, we defer
to the fast pose classifier's honest motion label (standing / walking / …).

Threading
---------
Inference runs on a dedicated worker thread fed by a bounded, drop-oldest
queue. The pipeline thread submits a clip and immediately continues with the
last cached prediction — the video feed never waits on the model.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from config import settings

from .object_context import ObjectContextDetector
from .target_vocabulary import resolve_indices


@dataclass
class ActionPrediction:
    label: str
    confidence: float
    # Context for the caption, never used to decide the label itself:
    # the object in hand when it agrees with the activity, and the posture
    # when one was actually observed (see update()).
    detail: Optional[str] = None
    posture: Optional[str] = None


_CLIP_LEN = 16
_INPUT_HW = (224, 224)

# --- Output gates ---
# coverage: fraction of total probability mass landing in our target vocabulary.
#   Low coverage means the scene is something we don't track (a sport, an
#   animal, an empty room) — defer to pose rather than forcing a label.
_MIN_COVERAGE = 0.22

# Confidence is handled in two tiers rather than one hard cut:
#
#   STRONG  — a single prediction this confident is surfaced immediately.
#   VOTE    — weaker predictions go into a short per-person vote buffer. If
#             the same category wins repeatedly, that consistency is itself
#             evidence: four independent 0.38 predictions all saying "cooking"
#             is a much stronger signal than the number 0.38 suggests.
#
# A single hard threshold throws that consistency away and was the direct
# cause of correct "cooking" labels being rejected at 0.37-0.39.
# Tuned for the s3d backend, which is ~7x faster than VideoMAE and therefore
# delivers roughly twice as many predictions in the same window (measured:
# 6-9 per clip vs 3-5). That extra throughput is spent on STRICTER consensus
# — a wider window needing more agreeing votes — which filters s3d's noisier
# per-prediction output without costing responsiveness.
# No reported confidence ever exceeds this. A statistical classifier
# claiming 100% certainty is always wrong, and it poisons trust in the
# honest numbers next to it.
_MAX_REPORTED_CONF = 0.95
_CONF_STRONG = 0.60        # a lone prediction must be quite strong to stand alone
_CONF_VOTE = 0.15          # floor to even enter the vote buffer
_VOTE_WINDOW = 6           # how many recent predictions to consider
_VOTE_MIN_HITS = 3         # need at least this many agreeing votes

# Evidence = SUM of the agreeing votes' confidences, not their mean.
#
# Mean was the wrong statistic and it cost real accuracy: on a kitchen clip
# the model returned "cooking" 12 times out of 12 at 0.19-0.58 confidence,
# but the mean sat right at the threshold so nothing was ever surfaced.
# Summing rewards agreement the way it should — repeated independent
# agreement IS evidence, even when each individual vote is modest:
#
#     6 votes x 0.30 = 1.80  -> confident (many weak agreeing observations)
#     3 votes x 0.28 = 0.84  -> confident (enough independent agreement)
#     2 votes x 0.30 = 0.60  -> not yet   (too few to trust)
#     2 votes x 0.55 = 1.10  -> confident (few but strong)
#
# Set to 0.75 rather than 1.0 because track IDs churn: a short looping clip
# reassigns a new ID each loop, so a single person rarely accumulates more
# than 3-4 votes before their buffer resets. Requiring 1.0 meant correct,
# unanimous predictions were discarded purely because the tracker renumbered
# the subject.
#
# Rescaled to 0.55 when confidence became coverage-tempered: tempering lowers
# every confidence by roughly 30%, so an un-rescaled threshold silently
# rejected everything. Thresholds and the confidence scale they act on have to
# move together.
#
# Raised again, 0.55 -> 0.70, together with _VOTE_MIN_HITS 2 -> 3, after
# measuring exactly the failure a client demo surfaced: a seated person
# drinking was read as "exercising" at 56% within the first few seconds,
# purely because two modest, UNRELATED votes (2 x ~0.28) happened to agree
# before the real "drinking" evidence had time to accumulate. Reference-clip
# testing (scripts/eval_candidate_clips.py) shows the same shape of noise —
# a single stray 0.36 "cleaning" read appearing between the correct
# "smoking"/"drinking" votes on otherwise-clean footage. Two weak agreeing
# votes is not enough to safely commit to; three is a meaningfully higher bar
# for very little added latency (one more ~1s inference cycle).
_VOTE_MIN_EVIDENCE = 0.70

# Hysteresis: the bar to REPLACE a label that is already showing and still
# fresh is higher than the bar to establish one in the first place.
#
# Without this, "flicker" is structural, not incidental: _tally() decides
# per-category, blind to what is currently displayed, so a transient category
# only needs to clear its OWN low bar to overwrite a correct answer that's
# already on screen. This is the direct fix for "before the precise one it is
# giving less confident other labels" — a specific category now needs to
# clearly outweigh the incumbent, not merely out-vote its own history.
# Applies only when the incoming category differs from what's currently
# shown; reinforcing the same label always uses the normal bar above.
_VOTE_SWITCH_EVIDENCE = 1.05

# Scene-level pool: wider window, and a higher bar since it mixes observations
# across track IDs rather than being about one specific person.
_SCENE_VOTE_WINDOW = 10
_SCENE_MIN_HITS = 5
_SCENE_MIN_EVIDENCE = 1.10
_SCENE_SWITCH_EVIDENCE = 1.50

_DEEP_CACHE_SECONDS = 6.0

# Display hold: how long a track's OWN confirmed verdict keeps showing with no
# supporting vote at all before we give up on it.
#
# This used to be _DEEP_CACHE_SECONDS too, and that was the direct cause of a
# seated smoker reading "standing 89%" about half the time. Measured live on
# demo-smoking: inferences land every ~3.6s, but 46% are coverage REJECTs and
# a third of the rest are stray categories, so a vote for the confirmed label
# only arrives every ~11s. A 6s window therefore expired between votes, the
# display dropped to the pose fallback, and came back on the next vote —
# smoking 39% / standing 61% on a clip where nothing changed.
#
# "Fresh enough to keep showing" and "fresh enough to hand to someone else"
# are different questions. A confirmed label on the same track now holds
# until a different category clears the switch bar (_VOTE_SWITCH_EVIDENCE),
# the track is reset, or this long limit passes with no vote for it. The
# scene verdict inherited by a just-renumbered track still uses the short
# _DEEP_CACHE_SECONDS, because that one crosses identities.
_DISPLAY_HOLD_SECONDS = 30.0

# Ceiling on a "standing" that comes from stillness alone.
#
# The pose engine only ever emits "standing" from _motion_label — i.e. no
# posture rule fired and the box is not moving. Its confidence measures how
# STILL the person is (0.02 body-heights/s -> 0.89), not how upright, so a
# seated, close-framed person scored higher than any real activity label on
# screen. The label is kept (a tracked person is never unlabelled, and it is
# right for someone genuinely standing); the number is capped below the range
# a confirmed activity reports (consensus lands at ~0.47-0.60) so it can no
# longer read as the system's most certain answer. Walking / running / sitting
# are untouched.
_STILLNESS_CONF_CAP = 0.40

# Deep-model state key used whenever exactly one person is in shot.
#
# In that case the orchestrator hands us the WHOLE FRAME, not a crop, so the
# clip buffer, the votes and the confirmed label describe the scene — they do
# not depend on which number ByteTrack happened to give the person. Keying
# them by track ID anyway threw all of it away on every renumbering, and on a
# clip whose tracks never survive 16 consecutive detections the deep model
# never ran at all. Measured live on demo-cycling: 11 IDs in 30 s, zero
# completed inferences, "detecting…" on 97% of labelled frames — on a clip
# the model labels correctly at 0.94 when it is allowed to look.
#
# ByteTrack IDs start at 1, so 0 can never collide with a real track.
_SCENE_KEY = 0

# If the single-person scene has been EMPTY this long, whoever appears next
# is not assumed to be continuing the same activity: the shared state is
# dropped. Long enough to ride out detector dropouts and a short clip
# looping; short enough that a different person walking in starts clean.
_SCENE_GAP_RESET_S = 5.0

# Motion labels that are real evidence in their own right. A box that is
# measurably travelling is walking/running whether or not the deep model has
# had its first look, so these are shown instead of "detecting…". (Stillness
# is not evidence of anything — "standing" stays behind the gate and capped.)
_MOTION_EVIDENCE = {"walking", "running"}
# ...but only once the track is old enough for the speed to be a measurement
# rather than two jittery boxes.
_MOTION_MIN_UPDATES = 4

# How long the scene verdict stays available to a track after the shot
# stopped being single-person. A second box that flickers in and out (a
# reflection, a ghost detection, someone crossing the background) flips the
# shot between the two keying modes; without a short handoff the confirmed
# label vanished every time it did (measured: demo-using-computer dropped
# from 86% to 50% correct). Deliberately short — a genuine newcomer should
# not wear the previous occupant's activity for long.
_SCENE_HANDOFF_S = 2.0

# Portrait frames (phone video) are cropped to a square around the person
# before the resize instead of being squashed.
#
# A plain resize of a 478x850 frame to 224x224 stretches the subject ~1.8x
# sideways — a distortion Kinetics models never saw. Measured over live-like
# 16-frame windows (correct windows / total, person-centred square vs squash):
#
#     demo-drinking (720x1280)        6/8  vs 3/8
#     phone clip, typing on a laptop  7/7  vs 5/7   (coverage 0.89 vs 0.52)
#
# Landscape frames are deliberately left alone: the same crop there helped
# one clip (eating 1/10 -> 6/10) and broke another (smoking 3/13 -> 0/13,
# the cigarette hand sits at the frame edge), so there is no honest case for
# changing them.
_PORTRAIT_RATIO = 1.2

# "exercising" is not shown over a person the pose engine has read as seated
# for this long (video seconds).
#
# Kinetics has no class for "sitting in an office chair", so for a seated
# person the nearest in-vocabulary neighbours are gym classes: measured on a
# phone clip of a seated man drinking from a bottle, the model answered
# "bench pressing" / "exercising arm" on 8 of 9 windows at up to 0.62. The
# posture read is the stronger evidence there. The duration matters: the
# bottom of a squat also reads "sitting", but for well under a second.
_SEATED_VETO = {"exercising"}
_SEATED_VETO_S = 2.0

# Kinetics classes that are only possible with the body horizontal. A vote
# for one of these is discarded when the skeleton plainly shows an upright
# torso — the two cannot both be true, and the skeleton is the direct
# measurement.
#
# Found on a phone clip of a man sitting down onto an office chair: the model
# answered "situp" on every window (0.49-0.87, coverage up to 0.97), and
# "bench pressing" for a seated man drinking. High confidence is no
# protection here; the scene simply has no closer class.
_HORIZONTAL_ONLY = {"situp", "push up", "bench pressing"}
# Partner dances need a partner. Measured on a T4: a lone runner (low-res
# demo-basic-run) read as "tango dancing" 6 times and showed "dancing" 37% of
# the time, halving its correct "running". With one person in shot these
# classes are not evidence of anything, the same way _HORIZONTAL_ONLY works.
_PARTNER_ONLY = {"tango dancing", "salsa dancing", "swing dancing"}
# Shoulder-to-hip drop, as a fraction of box height, above which the torso
# counts as upright. An upright adult measures 0.28-0.32 (see _is_crawling in
# action_classifier.py); a genuinely horizontal one is under 0.15.
_UPRIGHT_TORSO = 0.24
# ...and the torso line must be at least this steep (rise over run; 2.0 is
# about 63 degrees from horizontal).
_UPRIGHT_SLOPE = 2.0

# Walking / running read from a box whose skeleton is unreadable is not
# shown; it is reported as the (capped) stillness label instead.
#
# An unreadable skeleton here almost always means a partial body — hands at a
# pot, a waist-up close-up, a figure in a dark crowd — and a partial box
# changes size and position as limbs enter and leave it. That jitter was
# being measured as travel. On the 26 labelled clips it was the largest
# single source of wrong labels on screen: "walking"/"running" at 0.48-0.62
# over people who never left their seat (demo-eating 74% of the time,
# cooking and eating close-ups 5-11%). People who really are walking have
# readable skeletons — 95-100% of samples on every walking clip measured,
# including small figures on street-surveillance — so little is lost.
_UNVERIFIED_MOTION = {"walking", "running"}

# Confidence reported for a posture read from appearance (clip_posture.py).
# Above the stillness cap, since it is a real observation, but below what
# the knee-geometry rule reports (0.75).
_LOOKED_POSTURE_CONF = 0.70
# An object seen in the person's hands names the activity's object in the
# caption ("drinking from a cup") for this long after it was last seen.
_OBJECT_DETAIL_S = 6.0
# _lower_body_hidden_in_frame(): knees below this visibility count as hidden;
# a box no taller than this many widths is head-and-torso only (a standing
# person in full view measures 2.5-3); and a box whose bottom is within the
# last few percent of the frame is cut off by the frame, not by furniture.
_HIDDEN_KNEE_VIS = 0.30
_SEATED_MAX_ASPECT = 1.5
_CUT_BY_FRAME = 0.92

# Per-track state is dropped once a track has not been seen for this long.
#
# Nothing ever removed it: the orchestrator resets the smoother and the event
# log when a track disappears, but never the classifier, so every ID the
# tracker had ever issued kept its 16-frame clip buffer (2.4 MB) until the
# next source switch. A demo that loops for minutes never notices; a camera
# that runs for days in a busy scene does. Far longer than the display hold,
# so nothing that could still be shown is discarded.
_STALE_TRACK_S = 120.0
_PRUNE_EVERY_S = 10.0

# The pose engine's "push-up" and the deep model's "push up" are the same
# activity, and the deep one is reported under the "exercising" category.
# Left as two names, one person doing push-ups alternated between
# "exercising" and "push-up" on screen depending on which layer answered
# that frame. One activity, one name: the category the rest of the
# vocabulary already uses.
_POSE_TO_CATEGORY = {"push-up": "exercising"}

# Minimum wall-clock gap between deep inferences, across ALL people.
#
# This is the single most important knob for perceived speed. Measured on this
# machine: the tracker runs at 51 ms/frame (~20 FPS) alone, but 647 ms/frame
# (~1.5 FPS) while s3d inference is running — a 5.3x collapse, because both
# saturate all CPU cores and thrash.
#
# Pacing the worker bounds that contention to a duty cycle: with a ~310 ms
# inference and a 1.2 s floor, the deep model occupies the CPU ~25% of the
# time instead of ~100%, so the video stays smooth. We still get a fresh
# prediction roughly once a second, which is faster than any human needs to
# see an activity label update.
_MIN_INFER_INTERVAL = float(
    getattr(settings.model, "deep_min_interval_s", 2.5)
)
# Submit a clip every N frames per person. 8 gives a fresh deep prediction
# roughly twice a second per person — responsive enough that a label appears
# within ~1s of an activity starting, while still ~half the inference volume
# of running every frame.
_STRIDE_FRAMES = 8
_MAX_QUEUED_JOBS = 4

# Pose labels confident enough to skip the deep model entirely. These are
# activities the pose geometry identifies directly and reliably, so spending
# 150-300 ms of CPU to second-guess them is wasted work.
#
# Everything NOT in this set — including postures like sitting / bending /
# kneeling / lying — still runs the deep model, because a *seated* person is
# precisely who might be eating, writing, reading, or using a computer.
# Posture alone cannot distinguish those; only the deep model can.
# A pose label is decisive ONLY when its geometry is unambiguous and no other
# target category shares that geometry.
#
# Deliberately NOT decisive:
#   reading / writing / using computer / eating  — all "hands in front, head
#       tilted down". Pose cannot separate them; only the deep model can.
#   drinking / using phone  — both "hand raised to face", and eating shares it.
# There are two tiers, because "pose wins outright" was doing real damage.
#
# It used to be one flat set that returned the pose label unconditionally AND
# suppressed deep submission. That combination is a latch: one bad geometric
# read stops the deep model from ever running again for that person, so
# nothing can correct it. Measured on the demo set (scripts/eval_offline.py):
#
#   demo-cycling   deep said cycling 0.94/cov 0.98 -> overridden to "crawling"
#                  (a cyclist leans forward: flat torso, hands and knees fwd)
#   demo-writing   deep said using-computer 1.00 x8 -> overridden to "push-up"
#
# Both were right and both were thrown away for a rule with no scene context.
#
# ABSOLUTE wins no matter what. Only falling qualifies: it is safety-critical,
# so a miss costs more than a false alarm, and it is a motion spike the deep
# model has no comparable class for.
_POSE_ABSOLUTE = {"falling"}

# PREFERRED wins only when the deep model has no confident fresh opinion.
# These geometries are real but not exclusive — plenty of in-vocabulary
# activities produce a horizontal torso.
#
# (waving / clapping used to sit here and are now gone entirely: the wrist
# rules that produced them were removed after measuring typing->clapping 34/36
# and sitting->waving 32/36. Nothing can emit those labels any more.)
_POSE_PREFERRED = {
    "push-up",    # torso horizontal, wrists below shoulders, ankles level
    "crawling",   # flat torso with knees AND hands on the floor
}

_POSE_DECISIVE = _POSE_ABSOLUTE | _POSE_PREFERRED

# How long to withhold ANY posture/motion guess for a brand-new track ID
# before falling back to one.
#
# This exists for a specific, observed case: when ByteTrack assigns a fresh
# ID (a looping demo clip restarting, an occlusion, someone re-entering
# frame), the deep model has produced nothing for THAT id yet — but the pose
# fallback runs unconditionally and its own confidence formula can output a
# very high number (measured: "standing" at 93%) for simple stillness. That
# reads as "the system has decided" on screen when it has decided nothing at
# all; the client-visible symptom was a confident-looking wrong label on the
# very first frame of every new track.
#
# First version of this gated on a fixed wall-clock duration (4s: buffer-fill
# + one _MIN_INFER_INTERVAL cycle). Measured consequence on the eval suite:
# several reference clips are only 3-5s long in the first place, so a flat 4s
# timer could outlast the ENTIRE clip and show nothing but "detecting…" the
# whole time — worse than the guess it was replacing. A fixed duration is
# also just the wrong tool: how long the first real inference actually takes
# depends on CPU load and how many other people are in frame, not a constant.
#
# So this gates on an EVENT instead of a clock: has the deep model completed
# even one attempt at THIS track yet (see _infer_attempted, set in
# _worker_loop right after an inference call returns — successful, rejected
# on coverage, or otherwise, since a completed attempt is real information
# either way)? Before that, show "detecting…" honestly, for exactly as long
# as it genuinely takes. After that, a plain pose fallback is no longer
# "we haven't looked yet" — it's "we looked, and this is the honest read" —
# which is a legitimate answer to show, including on a 3-second clip.
_DETECTING = ActionPrediction(label="detecting…", confidence=0.0)


# Kinetics-400 normalization constants (used by the torchvision video models).
_KIN_MEAN = (0.43216, 0.394666, 0.37645)
_KIN_STD = (0.22803, 0.22145, 0.216989)


def _lower_body_hidden_in_frame(keypoints, bbox_xyxy, frame_hw) -> bool:
    """Shoulders in view, legs not, and the box stops well short of the
    frame's bottom edge: something INSIDE the picture — a desk, a chair
    back — is hiding the lower body. That is someone seated at a desk.

    Seen from behind in an office (the user's own camera), CLIP could not
    decide sitting vs standing (0.20-0.63), and the knee rule had no knees.
    The box shape settles it: a person standing in shot shows their legs or
    runs off the bottom of the frame; a short box ending mid-frame is a
    person cut off by furniture. The bottom-edge test is what keeps
    close-ups (a cook at a stove, someone brushing their teeth, all cut off
    by the frame itself) from being called seated.
    """
    if keypoints is None or frame_hw is None or len(keypoints) < 17:
        return False
    x1, y1, x2, y2 = bbox_xyxy
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    shoulders = min(keypoints[5][2], keypoints[6][2])
    knees = max(keypoints[13][2], keypoints[14][2])
    return (shoulders >= 0.5 and knees < _HIDDEN_KNEE_VIS
            and h / w <= _SEATED_MAX_ASPECT
            and y2 <= frame_hw[0] * _CUT_BY_FRAME)


def _torso_upright(keypoints, bbox_xyxy) -> bool:
    """True only when a clearly visible torso is clearly vertical."""
    if keypoints is None or len(keypoints) < 17:
        return False
    # COCO-17: shoulders 5/6, hips 11/12.
    if min(keypoints[5][2], keypoints[6][2], keypoints[11][2], keypoints[12][2]) < 0.5:
        return False
    shoulder_x = (keypoints[5][0] + keypoints[6][0]) / 2
    shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
    hip_x = (keypoints[11][0] + keypoints[12][0]) / 2
    hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
    drop, run = hip_y - shoulder_y, abs(hip_x - shoulder_x)
    # The torso LINE must be steep, not merely long: a plank on straight arms
    # has its shoulders well above its hips too, but mostly sideways of them.
    body_h = max(1.0, bbox_xyxy[3] - bbox_xyxy[1])
    return drop / body_h >= _UPRIGHT_TORSO and drop >= _UPRIGHT_SLOPE * run


class VideoMAEClassifier:
    """Restricted-vocabulary deep action classifier + fast pose fallback.

    Two interchangeable backends, both pretrained on Kinetics-400 so the
    target vocabulary in `target_vocabulary.py` applies to either:

        's3d'       torchvision S3D — 8.3M params, ~310 ms/clip on CPU
        'videomae'  VideoMAE-base   — 86.5M params, ~2260 ms/clip on CPU

    Measured on this machine, s3d is ~7x faster. Because the classifier's
    accuracy depends on accumulating several predictions (consensus voting),
    being 7x faster also makes it measurably *more accurate* in practice —
    it gets 7x more votes in the same wall-clock window.
    """

    _VMAE_CHECKPOINT = "MCG-NJU/videomae-base-finetuned-kinetics"

    def __init__(self, backend: str | None = None):
        from src.action_recognition.action_classifier import SkeletonHeuristicClassifier
        self._fallback = SkeletonHeuristicClassifier()
        # Appearance-based sitting/standing for when the joints can't say
        # (see clip_posture.py). Optional: absent CLIP, behaviour is unchanged.
        from .clip_posture import ClipPosture
        self._posture = ClipPosture()
        # Loaded unconditionally, independent of whether the deep scene model
        # below loads successfully — it degrades gracefully on its own
        # (`.available` is False if its weights are missing), same pattern as
        # every other optional layer in this class.
        self._object_ctx = ObjectContextDetector()
        self._clip_len = _CLIP_LEN
        self._stride = _STRIDE_FRAMES
        self._model = None
        self._backend = (backend or getattr(settings.model, "deep_backend", "s3d")).lower()

        try:
            import torch
            self._torch = torch
            model = None
            if self._backend == "videomae":
                try:
                    model, id2label = self._load_videomae()
                except Exception as e:
                    # Weights not downloaded on an offline box, say. S3D is
                    # bundled, so degrade to it rather than to pose-only.
                    print(f"[deep] VideoMAE unavailable ({e}); falling back to s3d",
                          flush=True)
                    self._backend = "s3d"
            if model is None:
                self._backend = "s3d"
                model, id2label = self._load_s3d()
            self._model = model
            self._id2label = id2label

            # Resolve our target vocabulary against the model's label space.
            allowed, idx_to_cat = resolve_indices(self._id2label)
            if not allowed:
                raise RuntimeError("target vocabulary resolved to zero classes")
            self._allowed = allowed
            self._idx_to_cat = idx_to_cat
            self._allowed_t = torch.tensor(allowed, dtype=torch.long, device=settings.device)

            categories = sorted(set(idx_to_cat.values()))
            print(f"[deep] backend={self._backend} on {settings.device}", flush=True)
            print(f"[deep] restricted vocabulary: {len(allowed)} Kinetics classes "
                  f"-> {len(categories)} categories: {', '.join(categories)}", flush=True)
        except Exception as e:
            print(f"[deep] unavailable, pose-only mode. ({e})", flush=True)
            self._model = None
            return

        # Per-person state, guarded by _lock (worker + pipeline threads share it)
        self._lock = threading.Lock()
        self._crops: dict[int, deque[np.ndarray]] = {}
        self._frame_counter: dict[int, int] = {}
        self._last_pred: dict[int, ActionPrediction] = {}
        self._last_pred_at: dict[int, float] = {}
        # When the confirmed label last got ANY accepted vote for it, whether
        # or not that vote re-cleared consensus — see _DISPLAY_HOLD_SECONDS.
        self._last_support_at: dict[int, float] = {}
        # Most recent object in hand per state key: (name, category, time).
        self._last_object: dict[int, tuple[str, str, float]] = {}
        self._observed_posture: Optional[str] = None
        # Updates seen per real track ID (the scene key never appears here).
        self._track_updates: dict[int, int] = {}
        # Wall-clock last-seen per state key, for pruning (see _STALE_TRACK_S).
        self._seen_at: dict[int, float] = {}
        self._pruned_at = time.time()
        # Last time the single-person scene key was fed — see _SCENE_GAP_RESET_S.
        self._scene_key_seen_at = 0.0
        # Video time at which each track's current unbroken "sitting" read
        # began — see _SEATED_VETO_S.
        self._sitting_since: dict[int, float] = {}
        self._votes: dict[int, deque[tuple[str, float]]] = {}
        # IDs the deep worker has completed at least one inference attempt
        # for — see the "detecting…" placeholder logic in update().
        self._infer_attempted: set[int] = set()
        # Has the model looked AT ALL, for ANY person, ever (this source)?
        #
        # Real bug this fixes: on a clip with unstable tracking, most
        # individual track IDs never survive the 16 CONSECUTIVE frames their
        # own clip buffer needs to submit at all — so `_infer_attempted`
        # legitimately never gets set for them. The existing scene-level
        # rescue (`self._scene_pred`, below) does not help either, because it
        # only sets once a REAL CATEGORY has been established — and a plain
        # "just sitting, no target activity" clip is mostly coverage REJECTs,
        # which carry real information (the model looked and found nothing
        # in-vocabulary) but never populate `_scene_pred`. Measured directly:
        # demo-basic-sit produced exactly one completed inference (a REJECT)
        # for the whole clip, and every churned track before and after it
        # showed "detecting…" — 138 of 210 raw predictions — because neither
        # rescue path recognized that the model HAD, in fact, looked.
        self._scene_attempted = False
        # Scene-level vote pool, shared across all track IDs.
        #
        # Per-person buffers are the right unit semantically, but they are
        # fragile in practice: ByteTrack reassigns IDs whenever a track is
        # lost and re-acquired (every loop of a short clip, or any occlusion),
        # which resets that person's evidence to zero. Measured on the kitchen
        # clip: the model said "cooking" 12/12 times, yet nothing surfaced,
        # because no single ID ever held more than 3 votes.
        #
        # The scene pool gives a newly-numbered track the benefit of what the
        # scene has consistently looked like. It is only consulted when the
        # person's own buffer is too thin to decide.
        self._scene_votes: deque[tuple[str, float]] = deque(maxlen=_SCENE_VOTE_WINDOW)
        # The scene verdict is cached at scene level, NOT under the track ID
        # that happened to trigger it — otherwise the very ID churn this is
        # meant to survive would hide it again from every newly-numbered track.
        self._scene_pred: Optional[ActionPrediction] = None
        self._scene_pred_at: float = 0.0

        # Generation counter: bumped on every source switch. A clip submitted
        # under generation N carries N with it; when the result comes back we
        # discard it unless the generation still matches. This is what stops
        # an inference that was already *running* when the source changed from
        # writing a label that belongs to the previous video.
        self._generation = 0

        self._jobs: "queue.Queue[tuple[int, int, list[np.ndarray]]]" = queue.Queue(
            maxsize=_MAX_QUEUED_JOBS
        )
        # Throttles the object-context layer to every other worker cycle —
        # see the comment at its call site in _worker_loop.
        self._obj_cycle = 0
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    # Backend loaders
    # ------------------------------------------------------------------
    def _load_s3d(self):
        """torchvision S3D, Kinetics-400. ~310 ms/clip on CPU."""
        from torchvision.models.video import S3D_Weights, s3d
        weights = S3D_Weights.KINETICS400_V1
        model = s3d(weights=weights).eval().to(settings.device)
        categories = weights.meta["categories"]
        id2label = {i: name for i, name in enumerate(categories)}
        # Pre-build normalization tensors once.
        t = self._torch
        self._mean = t.tensor(_KIN_MEAN, device=settings.device).view(1, 3, 1, 1, 1)
        self._std = t.tensor(_KIN_STD, device=settings.device).view(1, 3, 1, 1, 1)
        return model, id2label

    def _load_videomae(self):
        """HuggingFace VideoMAE-base. Much more accurate per-clip, but ~7x slower."""
        try:
            from transformers import (
                VideoMAEForVideoClassification,
                VideoMAEImageProcessor,
            )
        except ImportError as e:
            # transformers is an optional extra: it is large, and the default
            # s3d backend does not need it. Say so plainly rather than letting
            # a bare ImportError surface from three frames down.
            raise ImportError(
                "deep_backend is set to 'videomae', which needs the optional "
                "extras:\n\n    pip install -r requirements-optional.txt\n\n"
                "Or set `deep_backend: s3d` in config/model_config.yaml — it is "
                "the default, needs nothing extra, and is ~7x faster on CPU."
            ) from e
        self._processor = VideoMAEImageProcessor.from_pretrained(self._VMAE_CHECKPOINT)
        model = VideoMAEForVideoClassification.from_pretrained(self._VMAE_CHECKPOINT)
        model.eval().to(settings.device)
        t = self._torch
        self._vmae_mean = t.tensor(self._processor.image_mean, device=settings.device).view(1, 1, 3, 1, 1)
        self._vmae_std = t.tensor(self._processor.image_std, device=settings.device).view(1, 1, 3, 1, 1)
        return model, model.config.id2label

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def update(
        self,
        person_id: int,
        keypoints,
        bbox_xyxy: tuple[float, float, float, float],
        crop_bgr: Optional[np.ndarray] = None,
        timestamp: float | None = None,
        single_person: bool = False,
        frame_hw: tuple[int, int] | None = None,
    ) -> ActionPrediction | None:
        """The label for this person now, plus caption context.

        The label is decided exactly as before (_decide); this only attaches
        what the caption may add to it, and only when it is corroborated:
        the object detector saw an object of the SAME activity in the
        person's hands within the last few seconds, or a posture was
        observed (knees, appearance or hidden-legs geometry) rather than
        guessed from stillness.
        """
        self._observed_posture = None
        pred = self._decide(person_id, keypoints, bbox_xyxy, crop_bgr,
                            timestamp, single_person, frame_hw)
        if pred is None or self._model is None:
            return pred
        key = _SCENE_KEY if single_person else person_id
        detail = None
        with self._lock:
            seen = self._last_object.get(key)
        if (seen is not None and seen[1] == pred.label
                and time.time() - seen[2] < _OBJECT_DETAIL_S):
            detail = seen[0]
        posture = None if pred.label in ("sitting", "standing") else self._observed_posture
        return ActionPrediction(label=pred.label, confidence=pred.confidence,
                                detail=detail, posture=posture)

    def _decide(
        self,
        person_id: int,
        keypoints,
        bbox_xyxy: tuple[float, float, float, float],
        crop_bgr: Optional[np.ndarray] = None,
        timestamp: float | None = None,
        single_person: bool = False,
        frame_hw: tuple[int, int] | None = None,
    ) -> ActionPrediction | None:
        # Layer 1 — always cheap, always runs. Always per track: posture and
        # motion belong to a body, not to a scene.
        heur = self._fallback.update(person_id, keypoints, bbox_xyxy,
                                     crop_bgr, timestamp)
        # Tier membership is decided on the pose engine's own name; what is
        # REPORTED is the vocabulary's name for it (see _POSE_TO_CATEGORY).
        if (heur is not None and getattr(heur, "degraded", False)
                and heur.label in _UNVERIFIED_MOTION):
            heur = ActionPrediction(
                label="standing",
                confidence=min(heur.confidence, _STILLNESS_CONF_CAP),
            )
        # "standing" here only ever means "still, posture unknown". Ask how
        # the person looks instead — sitting at a desk with the legs out of
        # shot is the everyday case the joints cannot see.
        if heur is not None and heur.label == "standing" and crop_bgr is not None:
            key = _SCENE_KEY if single_person else person_id
            self._posture.submit(key, crop_bgr)
            hw = crop_bgr.shape[:2] if single_person else frame_hw
            seen = "sitting" if _lower_body_hidden_in_frame(keypoints, bbox_xyxy, hw)                 else self._posture.posture(key)
            if seen is not None:
                heur = ActionPrediction(label=seen, confidence=_LOOKED_POSTURE_CONF)
        if (heur is not None and heur.label in ("sitting", "standing")
                and heur.confidence >= _LOOKED_POSTURE_CONF):
            self._observed_posture = heur.label
        pose_label = heur.label if heur is not None else None
        if pose_label in _POSE_TO_CATEGORY:
            heur = ActionPrediction(label=_POSE_TO_CATEGORY[pose_label],
                                    confidence=heur.confidence)

        if self._model is None or crop_bgr is None or crop_bgr.size == 0:
            return heur

        # The clip buffer ALWAYS fills, regardless of what the pose said.
        # (A resize is microseconds; keeping the buffer temporally continuous
        # is what makes a 16-frame window available when we do want one.)
        try:
            src = crop_bgr
            ch, cw = crop_bgr.shape[:2]
            if single_person and ch > cw * _PORTRAIT_RATIO:
                # Square window, as wide as the frame, centred on the person
                # and clamped to the frame (see _PORTRAIT_RATIO).
                cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
                y0 = int(min(max(0, cy - cw / 2), ch - cw))
                src = crop_bgr[y0:y0 + cw]
            resized = cv2.resize(src, _INPUT_HW, interpolation=cv2.INTER_LINEAR)
        except cv2.error:
            return heur

        # Only an ABSOLUTE pose label suppresses deep submission. A merely
        # preferred one must NOT, or the deep model can never accumulate the
        # evidence needed to overturn it — that latch is what turned a cyclist
        # into a permanent "crawling".
        pose_absolute = pose_label in _POSE_ABSOLUTE
        pose_preferred = pose_label in _POSE_PREFERRED

        # Everything the deep model owns is keyed by the scene when the crop
        # IS the scene (see _SCENE_KEY), by the track otherwise.
        track_id = person_id

        # Sustained seated posture (see _SEATED_VETO_S). Video time when we
        # have it, so the answer does not depend on processing speed.
        clock = time.time() if timestamp is None else timestamp
        if pose_label == "sitting":
            seated = clock - self._sitting_since.setdefault(track_id, clock) >= _SEATED_VETO_S
        else:
            self._sitting_since.pop(track_id, None)
            seated = False
        if single_person:
            person_id = _SCENE_KEY

        with self._lock:
            if single_person:
                now = time.time()
                if (self._scene_key_seen_at
                        and now - self._scene_key_seen_at > _SCENE_GAP_RESET_S):
                    for d in (self._crops, self._frame_counter, self._last_pred,
                              self._last_pred_at, self._last_support_at,
                              self._votes):
                        d.pop(_SCENE_KEY, None)
                self._scene_key_seen_at = now
            self._track_updates[track_id] = self._track_updates.get(track_id, 0) + 1
            wall = time.time()
            self._seen_at[track_id] = wall
            self._seen_at[person_id] = wall
            if wall - self._pruned_at > _PRUNE_EVERY_S:
                self._pruned_at = wall
                for stale in [k for k, t in self._seen_at.items()
                              if wall - t > _STALE_TRACK_S]:
                    for d in (self._crops, self._frame_counter, self._last_pred,
                              self._last_pred_at, self._last_support_at,
                              self._votes, self._track_updates,
                              self._sitting_since, self._seen_at):
                        d.pop(stale, None)
                    self._infer_attempted.discard(stale)
                    self._fallback.reset(stale)
                    self._posture.forget(stale)
            track_age = self._track_updates[track_id]

            buf = self._crops.setdefault(person_id, deque(maxlen=self._clip_len))
            buf.append(resized)
            self._frame_counter[person_id] = self._frame_counter.get(person_id, 0) + 1
            counter = self._frame_counter[person_id]

            if (not pose_absolute
                    and len(buf) == self._clip_len
                    and counter % self._stride == 0):
                # crop_bgr/bbox_xyxy/keypoints ride along for the object-
                # context layer, which needs full resolution (small objects
                # like a phone or toothbrush disappear at the 224x224 the
                # scene classifier uses) and the person's actual geometry —
                # neither of which survive in `resized`.
                self._submit(person_id, list(buf), crop_bgr, bbox_xyxy, keypoints,
                             single_person)

            # A confirmed verdict on this track is HELD, not merely cached:
            # it keeps showing until out-voted, reset, or unsupported for
            # _DISPLAY_HOLD_SECONDS.
            deep = self._last_pred.get(person_id)
            deep_at = self._last_support_at.get(person_id, 0.0)
            fresh = deep is not None and (time.time() - deep_at) < _DISPLAY_HOLD_SECONDS

            # The shot just flipped between single- and multi-person keying:
            # carry the confirmed label across (see _SCENE_HANDOFF_S).
            if not fresh:
                if single_person:
                    other, ok = track_id, True
                else:
                    other = _SCENE_KEY
                    ok = (time.time() - self._scene_key_seen_at) < _SCENE_HANDOFF_S
                held = self._last_pred.get(other)
                if (ok and held is not None
                        and (time.time() - self._last_support_at.get(other, 0.0))
                        < _DISPLAY_HOLD_SECONDS):
                    deep, fresh = held, True

            # This track has no verdict of its own (usually because ByteTrack
            # just renumbered it) — inherit the scene's, if recent.
            if not fresh and self._scene_pred is not None:
                if (time.time() - self._scene_pred_at) < _DEEP_CACHE_SECONDS:
                    deep, fresh = self._scene_pred, True

            attempted = (
                person_id in self._infer_attempted
                or self._scene_attempted
            )

        # Resolution order:
        #   1. falling — always, it is the one label we refuse to lose.
        #   2. a confident fresh deep label — it has scene context, which the
        #      pose rules structurally do not.
        #   3. a preferred pose label — the deep model is unsure, geometry is
        #      the better guess. Real geometric evidence, shown immediately
        #      even for a brand-new track — the "haven't looked yet" gate
        #      below is about withholding UNEVIDENCED guesses, not this.
        #   4. the deep model has never once completed a look at this track
        #      (or, failing that, the scene) — say so honestly instead of a
        #      guess dressed up with a confident-looking number.
        #   5. the pose's honest posture/motion read — a legitimate answer
        #      once the deep model HAS had its look, whatever it concluded.
        if pose_absolute:
            return heur
        if fresh and deep is not None and seated and deep.label in _SEATED_VETO:
            return heur
        if fresh and deep is not None:
            if not pose_preferred or deep.confidence >= _CONF_STRONG:
                return deep
        if pose_preferred:
            return heur
        if not attempted:
            if (heur is not None and heur.label in _MOTION_EVIDENCE
                    and track_age >= _MOTION_MIN_UPDATES):
                return heur
            # A posture that was actually observed (knees, or appearance) is
            # evidence too; only bare stillness waits for the deep model.
            if (heur is not None and heur.label in ("sitting", "standing")
                    and heur.confidence >= _LOOKED_POSTURE_CONF):
                return heur
            return _DETECTING
        # Stillness is not evidence of posture — see _STILLNESS_CONF_CAP.
        if heur is not None and heur.label == "standing":
            return ActionPrediction(
                label=heur.label,
                confidence=min(heur.confidence, _STILLNESS_CONF_CAP),
            )
        return heur

    def reset(self, person_id: int) -> None:
        if self._model is not None:
            with self._lock:
                for d in (self._crops, self._frame_counter,
                          self._last_pred, self._last_pred_at,
                          self._last_support_at, self._votes,
                          self._track_updates, self._sitting_since):
                    d.pop(person_id, None)
                self._infer_attempted.discard(person_id)
        self._fallback.reset(person_id)

    def reset_all(self) -> None:
        """Drop ALL per-person state. Called when the video source changes.

        Without this, a 16-frame clip buffer can span two different videos
        (garbage input) and a cached prediction from the previous source
        stays 'fresh' long enough to be served for the new one — producing
        labels that lag a whole clip behind.
        """
        if self._model is not None:
            with self._lock:
                # Invalidate anything in flight: queued clips are drained, and
                # a clip already mid-inference is discarded on completion
                # because its generation no longer matches.
                self._generation += 1
                self._crops.clear()
                self._frame_counter.clear()
                self._last_pred.clear()
                self._last_pred_at.clear()
                self._last_support_at.clear()
                self._track_updates.clear()
                self._last_object.clear()
                self._seen_at.clear()
                self._sitting_since.clear()
                self._scene_key_seen_at = 0.0
                self._votes.clear()
                self._infer_attempted.clear()
                self._scene_attempted = False
                self._scene_votes.clear()
                self._scene_pred = None
                self._scene_pred_at = 0.0
                while True:
                    try:
                        self._jobs.get_nowait()
                    except queue.Empty:
                        break
        self._fallback.reset_all()
        if hasattr(self, "_posture"):
            self._posture.reset()

    # ------------------------------------------------------------------
    # Async worker
    # ------------------------------------------------------------------
    def _submit(self, person_id: int, clip: list[np.ndarray],
               crop_bgr: np.ndarray, bbox_xyxy: tuple[float, float, float, float],
               keypoints, single_person: bool = False) -> None:
        job = (self._generation, person_id, clip, crop_bgr, bbox_xyxy, keypoints,
               single_person)
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            try:
                self._jobs.get_nowait()          # drop oldest, keep latency bounded
            except queue.Empty:
                pass
            try:
                self._jobs.put_nowait(job)
            except queue.Full:
                pass

    def _worker_loop(self) -> None:
        last_infer_end = 0.0
        while True:
            gen, person_id, clip, crop_bgr, bbox_xyxy, keypoints, solo = self._jobs.get()
            # Cheap pre-check: skip work we already know is stale.
            if gen != self._generation:
                continue

            # Pace the worker so inference can't monopolise the CPU and starve
            # the video pipeline (see _MIN_INFER_INTERVAL).
            gap = time.time() - last_infer_end
            if gap < _MIN_INFER_INTERVAL:
                time.sleep(_MIN_INFER_INTERVAL - gap)
                if gen != self._generation:
                    continue

            try:
                category, confidence = self._infer(
                    clip, person_id, upright=_torso_upright(keypoints, bbox_xyxy), solo=solo)
                last_infer_end = time.time()
            except Exception as e:
                print(f"[deep] inference error: {e}", flush=True)
                continue

            # Object-context: a second, independent opinion from the same
            # frame the scene classifier just used. Piggy-backs on the deep
            # model's own pacing cadence rather than scheduling its own timer
            # — a second, uncoordinated background loop would reopen the
            # exact CPU-starvation problem this system already fixed once.
            #
            # Runs every OTHER cycle, not every one, purely to keep its CPU
            # cost down — object presence is slow-changing by nature (a
            # laptop doesn't leave the desk between one look and the next),
            # so skipping every other look costs little real signal.
            #
            # NOTE: this throttle was originally written to explain away a
            # regression (demo-basic-sit/walk/run failing) that turned out to
            # have a different, unrelated cause — see `_scene_attempted`
            # above. Confirmed directly: the same failure reproduced with
            # object-context fully disabled. The throttle is kept anyway
            # because halving a real, measured CPU cost for equivalent signal
            # is worth doing on its own merits, but it is not the fix for
            # that bug — do not remove `_scene_attempted` on the assumption
            # this comment's old reasoning still applies.
            self._obj_cycle += 1
            obj_result = None
            if self._object_ctx.available and self._obj_cycle % 2 == 0:
                try:
                    obj_result = self._object_ctx.infer(crop_bgr, bbox_xyxy, keypoints)
                except Exception as e:
                    print(f"[objects] inference error: {e}", flush=True)

            # A completed attempt, win or lose — this is what update() checks
            # to decide whether to show "detecting…" or fall back to pose.
            # Marked even on a coverage REJECT (category is None below): the
            # model genuinely looked and found nothing in-vocabulary, which is
            # real information, not "hasn't looked yet".
            with self._lock:
                self._infer_attempted.add(person_id)
                self._scene_attempted = True

            # Deep-scene and object-context are two independent opinions on
            # the same moment; each goes through the identical vote/tally
            # path below (_process_vote), so object evidence is subject to
            # the exact same evidence bar and switch-hysteresis as a scene
            # classifier vote — no separate, weaker-tested decision surface.
            #
            # Processed deep-first, object-second: object evidence is meant
            # to corroborate or tie-break, not lead. Doing it in this order
            # means an object vote that merely reinforces what the deep model
            # just said flows through as simple confirmation, while an object
            # vote that disagrees still has to clear the full switch bar.
            with self._lock:
                if gen != self._generation:      # source may have switched mid-inference
                    continue
                if category is not None and confidence >= _CONF_VOTE:
                    self._process_vote(person_id, category, confidence)
                if obj_result is not None:
                    self._process_vote(person_id, obj_result.label, obj_result.confidence)
                    name = getattr(obj_result, "detail", None)
                    if name:
                        self._last_object[person_id] = (name, obj_result.label, time.time())

    def _process_vote(self, person_id: int, category: str, confidence: float) -> None:
        """Append one (category, confidence) observation and re-tally.

        Caller must hold `self._lock`. Shared by both the deep scene
        classifier and the object-context layer so a vote from either source
        is bound by identical rules — same evidence thresholds, same
        switch-hysteresis against whatever is already showing.
        """
        votes = self._votes.setdefault(person_id, deque(maxlen=_VOTE_WINDOW))
        votes.append((category, confidence))
        self._scene_votes.append((category, confidence))

        # The hysteresis anchor: the last thing this person's track was
        # confidently called, PERIOD — not gated by _DEEP_CACHE_SECONDS.
        #
        # First version of this fix gated `current` by that same freshness
        # window (used elsewhere to decide what to DISPLAY), and it
        # measurably did nothing: live-tested on the drinking clip,
        # "drinking" kept alternating with "eating"/"smoking" exactly as
        # before. Cause: the real gap between inferences for one tracked
        # person is close enough to the 6s cache window that the anchor had
        # usually already gone "stale" (-> None) by the time the next tally
        # ran, so every switch was silently evaluated as if nothing were
        # showing yet.
        #
        # Hysteresis and "what's fresh enough to display right now" are
        # different questions. `_last_pred` is never cleared by mere
        # staleness anyway (only by reset()/reset_all()), so using it
        # directly gives an anchor that lasts as long as the track does,
        # independent of inference cadence.
        #
        # Falls back to the scene verdict when this track has none of its
        # own: a just-renumbered ID (ByteTrack churn) starts with an empty
        # vote buffer, and without this fallback that empty buffer would let
        # it bypass hysteresis entirely — a single confident-but-wrong read
        # on a brand new ID could momentarily override an answer the whole
        # scene had already settled on.
        current = self._last_pred.get(person_id) or self._scene_pred
        surfaced = self._tally(votes, category, confidence, current=current)
        if surfaced is not None:
            self._last_pred[person_id] = surfaced
            self._last_pred_at[person_id] = time.time()

        # Any accepted vote for the label already showing keeps the display
        # hold alive, even one too weak to re-clear consensus by itself.
        held = self._last_pred.get(person_id)
        if held is not None and held.label == category:
            self._last_support_at[person_id] = time.time()

        # Independently maintain the scene-level verdict, readable by any
        # track ID including ones that don't exist yet.
        scene = self._tally_scene(category, current=self._scene_pred)
        if scene is not None:
            self._scene_pred = scene
            self._scene_pred_at = time.time()

    def _tally_scene(self, category: str, current: ActionPrediction | None) -> ActionPrediction | None:
        """Decide from the scene-wide pool when one track lacks its own evidence.

        Deliberately stricter than the per-person rule (more hits, more
        evidence) because this mixes observations across identities — it should
        only fire when the scene has been unambiguous for a while.
        """
        matching = [c for (lbl, c) in self._scene_votes if lbl == category]
        if len(matching) < _SCENE_MIN_HITS:
            return None
        evidence = sum(matching)
        # See _tally's docstring: replacing an already-fresh, different
        # verdict needs to clearly outweigh it, not just clear its own bar.
        needed = (
            _SCENE_SWITCH_EVIDENCE
            if current is not None and current.label != category
            else _SCENE_MIN_EVIDENCE
        )
        if evidence < needed:
            return None
        mean_conf = evidence / len(matching)
        bonus = min(0.30, 0.06 * len(matching))
        reported = min(_MAX_REPORTED_CONF, mean_conf + bonus * (1.0 - mean_conf))
        print(f"[deep] SCENE-CONSENSUS {category:<16} "
              f"{len(matching)}/{len(self._scene_votes)} votes "
              f"evidence={evidence:.2f} -> conf={reported:.2f}", flush=True)
        return ActionPrediction(label=category, confidence=round(reported, 3))

    @staticmethod
    def _tally(votes, category: str, confidence: float,
               current: ActionPrediction | None = None) -> ActionPrediction | None:
        """Decide whether the evidence so far justifies surfacing a label.

        `current` is whatever is already displayed for this person AND still
        fresh (None otherwise) — see the call site. It exists to fix a real
        flicker bug: this function used to decide per-category with no idea
        what was already on screen, so a transient reading only had to clear
        its OWN low bar to overwrite a correct answer that got there first.
        Now replacing a different, still-fresh label costs more evidence than
        establishing one from a blank slate.
        """
        # Tier 1 — a single strong prediction stands on its own, but ONLY when
        # it is establishing a fresh read or reinforcing what's already shown.
        #
        # This used to fire regardless of `current`, on the reasoning that
        # every misfire observed in testing sat below _CONF_STRONG. That
        # reasoning held for the clips tested at the time, but real footage
        # broke it: on the wine-glass drinking clip, "drinking" and "eating"
        # each independently clear 0.60 on different frames (a sip and a
        # taste look alike to this model), so the display alternated between
        # two individually "confident" answers — visible in the client demo
        # as drinking flickering to "eating 85%" and back.
        #
        # A genuine SWITCH away from a different, still-fresh label now has to
        # go through Tier 2's evidence accumulation below, same as a weak
        # reading does. The trade-off is honest: switching categories takes a
        # few seconds longer, in exchange for not alternating between two
        # answers that are each momentarily "sure".
        if confidence >= _CONF_STRONG and (current is None or current.label == category):
            return ActionPrediction(label=category,
                                    confidence=min(_MAX_REPORTED_CONF, confidence))

        # Tier 2 — accumulated evidence across recent predictions.
        matching = [c for (lbl, c) in votes if lbl == category]
        if len(matching) < _VOTE_MIN_HITS:
            return None

        evidence = sum(matching)
        needed = (
            _VOTE_SWITCH_EVIDENCE
            if current is not None and current.label != category
            else _VOTE_MIN_EVIDENCE
        )
        if evidence < needed:
            return None

        # Reported confidence must stay honest. Repeated agreement genuinely
        # raises certainty, but four votes averaging 0.25 is NOT 94% certain —
        # an earlier formula claimed exactly that, which is the kind of number
        # that destroys trust in the whole system.
        #
        # So: start from the mean of the agreeing votes and close part of the
        # remaining gap to 1.0, proportional to how many independent
        # observations agreed. The bonus saturates, so weak-but-unanimous
        # reads as "moderately confident", never as near-certain.
        mean_conf = evidence / len(matching)
        agreement_bonus = min(0.35, 0.08 * len(matching))
        reported = min(_MAX_REPORTED_CONF,
                       mean_conf + agreement_bonus * (1.0 - mean_conf))
        print(f"[deep] CONSENSUS {category:<16} {len(matching)}/{len(votes)} votes  "
              f"evidence={evidence:.2f} -> conf={reported:.2f}", flush=True)
        return ActionPrediction(label=category, confidence=round(reported, 3))

    # ------------------------------------------------------------------
    # Restricted-vocabulary inference
    # ------------------------------------------------------------------
    def _infer(self, clip_bgr: list[np.ndarray], person_id: int = -1,
               upright: bool = False, solo: bool = False):
        torch = self._torch
        rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in clip_bgr], axis=0)

        with torch.no_grad():
            if self._backend == "s3d":
                # (T,H,W,C) uint8 -> (1,C,T,H,W) normalized float. Contiguous,
                # so the real call matches the warm-up's memory layout: with
                # cuDNN autotuning a permuted (strided) input is a "new" shape
                # and its first call paid ~4 s of tuning on a T4.
                x = torch.from_numpy(rgb).to(settings.device).float().div_(255.0)
                x = x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
                x = (x - self._mean) / self._std
                logits = self._model(x)[0]
            elif rgb.shape[1:3] == _INPUT_HW:
                # Frames are already 224x224, where the HF processor's resize
                # and crop are identities: rescale + normalise on the GPU
                # instead. Verified bit-identical (max abs diff 0.0) to the
                # processor, which on the CPU cut playback to 16 fps on a T4.
                x = torch.from_numpy(rgb).to(settings.device).float().div_(255.0)
                x = x.permute(0, 3, 1, 2).unsqueeze(0).contiguous()     # (1,T,C,H,W)
                x = (x - self._vmae_mean) / self._vmae_std
                logits = self._model(pixel_values=x).logits[0]
            else:
                inputs = self._processor(list(rgb), return_tensors="pt")
                inputs = {k: v.to(settings.device) for k, v in inputs.items()}
                logits = self._model(**inputs).logits[0]

            full_probs = torch.softmax(logits, dim=-1)

            # How much belief lands inside our target vocabulary at all?
            allowed_probs = full_probs[self._allowed_t]
            coverage = float(allowed_probs.sum())

            # Renormalize within the vocabulary and pick the winner.
            restricted = allowed_probs / allowed_probs.sum()
            best_pos = int(torch.argmax(restricted))
            in_vocab_conf = float(restricted[best_pos])

            # Temper by coverage before reporting.
            #
            # Renormalizing inside an 11-category vocabulary can return 1.0
            # whenever one in-vocabulary class dominates the others — even if
            # only 38% of the model's total belief was in our vocabulary at
            # all. Reporting that as "100% confident" is false: it means
            # "of the things we track, definitely this one", not "definitely
            # this action". Scaling by coverage makes high confidence require
            # BOTH a clear winner AND a scene that really is one of ours.
            confidence = in_vocab_conf * (0.5 + 0.5 * coverage)

        class_idx = self._allowed[best_pos]
        category = self._idx_to_cat[class_idx]
        raw = self._id2label[class_idx]

        # The scene isn't one of our tracked activities at all.
        if coverage < _MIN_COVERAGE:
            # Report what the model ACTUALLY believes. Without this, a
            # low-coverage rejection is indistinguishable from a broken
            # pipeline — you can't tell "model is confused" from "model
            # never ran", and that ambiguity wastes hours.
            with torch.no_grad():
                top3 = torch.topk(full_probs, 3)
            actual = ", ".join(
                f"{self._id2label[int(i)]}={float(p):.2f}"
                for p, i in zip(top3.values, top3.indices)
            )
            print(f"[deep] REJECT (coverage  ) would-be={category:<16} "
                  f"conf={confidence:.2f} coverage={coverage:.2f}", flush=True)
            print(f"[deep]   model actually sees: {actual}", flush=True)
            return None, None

        # In-vocabulary, but physically contradicted by the skeleton
        # (see _HORIZONTAL_ONLY). Still a completed look, like a coverage
        # reject — just not one that gets a vote.
        if upright and raw in _HORIZONTAL_ONLY:
            print(f"[deep] REJECT (posture  ) would-be={category:<16} "
                  f"conf={confidence:.2f} coverage={coverage:.2f}  "
                  f"(raw='{raw}' needs a horizontal body; torso is upright)",
                  flush=True)
            return None, None

        # A partner dance with one person in shot (see _PARTNER_ONLY).
        if solo and raw in _PARTNER_ONLY:
            print(f"[deep] REJECT (partner  ) would-be={category:<16} "
                  f"conf={confidence:.2f} coverage={coverage:.2f}  "
                  f"(raw='{raw}' needs two people; one in shot)", flush=True)
            return None, None

        # flush=True matters: without it stdout is block-buffered when the
        # process is piped to a file, and log lines land long after the fact.
        # Human-readable line for eyeballing the log...
        print(f"[deep] P{person_id:<4} {category:<16} conf={confidence:.2f} "
              f"coverage={coverage:.2f}  (raw='{raw}')", flush=True)
        # ...plus a machine-readable one for tooling. Keeping these separate
        # matters: scraping the prose line broke the validation harness three
        # times during development (once per cosmetic log tweak), each time
        # producing a convincing but entirely false "accuracy = 0%" result.
        print(f"DEEPPRED\t{person_id}\t{category}\t{confidence:.4f}\t{coverage:.4f}",
              flush=True)
        return category, confidence
