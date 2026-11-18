"""Turn per-person action labels into one plain-English caption.

Everything here is deterministic text assembly over labels the pipeline has
already decided — no model runs in this file. That is deliberate: the caption
can never claim more than the labels on screen do, so it adds no new way to
be wrong. The only judgement it makes is what to LEAVE OUT, which is where
"no noise" comes from:

  * a label must be present in most frames of a short window before the
    caption trusts it (_WINDOW_S), so a one-frame flicker never reaches the
    text;
  * a low-confidence label is not narrated at all (_MIN_CONFIDENCE) — the
    capped "standing 0.40" shown before an activity is confirmed would
    otherwise be read out as fact for a seated smoker;
  * the finished sentence itself must hold before it replaces the one on
    screen (_CAPTION_HOLD_S), so people stepping in and out of a corner of
    the frame do not make the subtitle jump.

Richer wording comes only from information that is itself corroborated:
the object in the person's hands when the object detector saw one of the
SAME activity ("drinking from a cup"), a posture that was observed rather
than guessed ("A seated person…"), and how long the main activity has
lasted ("…for about 3 minutes"). Each must hold for most of a few seconds
(_CONTEXT_WINDOW_S) before it is used, so it summarises the scene rather
than reporting the latest frame.
"""
from __future__ import annotations

import time
from collections import Counter, deque

# How each label reads after "is" / "are".
_PHRASES: dict[str, str] = {
    "eating": "eating",
    "drinking": "drinking",
    "smoking": "smoking",
    "cooking": "cooking",
    "cleaning": "cleaning",
    "using computer": "working on a computer",
    "using phone": "using a phone",
    "reading": "reading",
    "writing": "writing",
    "brushing teeth": "brushing their teeth",
    "exercising": "exercising",
    "cycling": "riding a bicycle",
    "driving": "driving",
    "swimming": "swimming",
    "dancing": "dancing",
    "sitting": "sitting",
    "standing": "standing",
    "walking": "walking",
    "running": "running",
    "falling": "falling",
    "falling down": "falling down",
    "making tea": "making tea",
    "taking tablets": "taking tablets",
    "bending": "bending down",
    "lying": "lying down",
    "kneeling": "kneeling",
    "crawling": "crawling",
}

# Specific activities are narrated ahead of plain posture/motion.
_POSTURE = {"sitting", "standing", "walking", "running", "bending", "lying",
            "kneeling", "crawling"}

_NUMBERS = ["no", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten"]

# Below this the label is a placeholder, not a finding: the capped stillness
# read (0.40), "detecting…" (0.0) and unverified motion all sit under it,
# while every confirmed activity reports 0.47 or more.
_MIN_CONFIDENCE = 0.45
# Everything is judged over this sliding window of processed frames, by
# MEDIAN count: how many people, and how many of them per activity.
#
# A first version followed each tracked person by ID and required their label
# to hold. Live, on a clip where the tracker renumbers people every second or
# so (demo-basic-run), that never settled: each new ID restarted the hold,
# the head-count flickered between 0, 1 and 2, and the caption sat on
# "Analysing the scene…" for 16 s. Counting labels per frame does not care
# what number the tracker gave anyone, and a median ignores single-frame
# blips in either direction.
#
# 2.0 s / 1.5 s rather than 1.5 s / 1.0 s: measured live on
# demo-using-computer, a second box that comes and goes (a ghost detection
# beside the typist) flipped the caption between "a person" and "two people"
# nine times in 30 s at the shorter settings.
_WINDOW_S = 2.0
_CAPTION_HOLD_S = 1.5

# Speed without noise: the window and hold above exist to stop flicker, and
# flicker is a property of the HEAD-COUNT and of labels that come and go —
# not of an activity that the classifier has just confirmed through its own
# vote. So the two are now judged separately:
#
#   * activities use a shorter window (_LABEL_WINDOW_S): by the time a
#     label reaches here it has already passed consensus voting upstream;
#   * a change that only ADDS information — "identifying the activity…"
#     becoming "cooking", or a place appearing — goes up after
#     _FAST_HOLD_S; anything that replaces or removes something already
#     on screen still needs the full _CAPTION_HOLD_S.
#
# Measured live on six clips: first real caption 4.1-6.7 s after the clip
# started with a single 1.5 s hold for everything.
_LABEL_WINDOW_S = 1.0
_FAST_HOLD_S = 0.4
# In a group the head-count itself wavers as people further back drop in and
# out of detection (classroom footage: 4 <-> 5 every few seconds). A change
# that is ONLY the number of people therefore needs to hold longer.
_COUNT_HOLD_S = 3.0
# ...and holding was not enough on its own: on a classroom clip the count
# swung 4 <-> 5 with a period of about 4 s, so a 3 s hold still let it
# through. In a group, the people the detector drops are the small ones at
# the back, so the recent MAXIMUM is the better estimate of who is there.
# The head-count therefore sticks at its recent high for this window.
# (Label counts were tried the same way and made titles worse: a wrong
# "running" on a seated student stayed on screen for the whole window.)
_PEOPLE_STICK_S = 8.0
# In a group, plain posture/motion is narrated only when it describes at
# least this share of the people. Measured on classroom footage, single
# students flickered between "walking", "running" and "bending down" from
# box jitter while seated; a whole class walking out still gets said.
_GROUP_POSTURE_SHARE = 0.5
_STARTING = "Analysing the scene…"

# Context (object, posture) is summarised over this longer window and must
# be the majority reading in it — a cup glimpsed once is not "drinking from
# a cup".
_CONTEXT_WINDOW_S = 5.0
_CONTEXT_SHARE = 0.5
# Posture is the shakier of the two (it can come from appearance alone), so
# it must be the reading in nearly every frame of the window before it is
# said, and a different posture must be just as clear to replace it.
# Measured live: at a 50% bar "seated" and "standing" took turns on the same
# diner every few seconds.
_POSTURE_SHARE = 0.9
# Once said, an object or posture stays while the activity it describes
# continues: a cup that drops out of view for a moment does not make the
# subtitle lose "from a cup" and then add it back (12 changes in 30 s on
# demo-drinking before this).
# Group posture clauses ("three are sitting") use a longer window too: in a
# class the seated count wobbled as students were detected and dropped.
_GROUP_POSTURE_WINDOW_S = 6.0
# An activity that pauses for less than this (a person briefly out of shot)
# keeps its running duration.
_DURATION_GAP_S = 4.0

# (activity, object) -> wording. Only pairs where the object genuinely
# names how the activity is done; anything else keeps the plain phrase.
_WITH_OBJECT: dict[tuple[str, str], str] = {
    ("drinking", "cup"): "drinking from a cup",
    ("drinking", "bottle"): "drinking from a bottle",
    ("drinking", "wine glass"): "drinking from a glass",
    ("using computer", "laptop"): "working on a laptop",
    ("using computer", "keyboard"): "typing on a keyboard",
    ("using computer", "mouse"): "working on a computer",
    ("using phone", "cell phone"): "using a phone",
    ("reading", "book"): "reading a book",
    ("eating", "fork"): "eating with a fork",
    ("eating", "spoon"): "eating with a spoon",
    ("eating", "bowl"): "eating from a bowl",
    ("eating", "sandwich"): "eating a sandwich",
    ("eating", "pizza"): "eating pizza",
    ("eating", "cake"): "eating cake",
    ("eating", "donut"): "eating a doughnut",
    ("eating", "hot dog"): "eating a hot dog",
    ("eating", "banana"): "eating a banana",
    ("eating", "apple"): "eating an apple",
    ("eating", "orange"): "eating an orange",
    ("brushing teeth", "toothbrush"): "brushing their teeth",
}
_POSTURE_ADJ = {"sitting": "seated", "standing": "standing"}
_FALL = {"falling", "falling down"}


def _count(n: int) -> str:
    return _NUMBERS[n] if n < len(_NUMBERS) else str(n)


def _phrase(label: str, details: dict | None) -> str:
    obj = (details or {}).get(label)
    return _WITH_OBJECT.get((label, obj), _PHRASES.get(label, label)) if obj else _PHRASES.get(label, label)


def _duration(seconds: float | None) -> str:
    if not seconds or seconds < 60:
        return ""
    m = int(seconds // 60)
    return f" for about {m} minute{'s' if m != 1 else ''}"


def compose_sentence(labels: list[str | None], scene: str | None,
                     details: dict | None = None, posture: str | None = None,
                     group_posture: str | None = None,
                     duration_s: float | None = None) -> str:
    """One sentence from the stable label of each person present.

    `labels` has one entry per person; None means "present, activity not yet
    known". `scene` is a phrase such as "in a kitchen", or None. Optional,
    corroborated context: `details` maps an activity to the object seen in
    hand, `posture` is a lone person's observed posture, `group_posture` a
    group's shared one, and `duration_s` how long the main activity has run.
    """
    where = f" {scene}" if scene else ""
    n = len(labels)
    if n == 0:
        # Say nothing rather than "no one is here": the detector misses small
        # or distant people (street-surveillance: people walking through the
        # whole clip, a person box on barely a quarter of frames), so an
        # empty frame is not evidence of an empty scene.
        return ""

    known = [l for l in labels if l]
    # A fall is an alert, said first and plainly, whatever else is going on.
    fallen = sum(1 for l in known if l in _FALL)
    if fallen:
        who = "a person has" if fallen == 1 else f"{_count(fallen)} people have"
        rest = n - fallen
        tail = f" {_count(rest).capitalize()} other{'s' if rest != 1 else ''} in view." if rest else ""
        return f"Alert: {who} fallen down{where}.{tail}"
    if n == 1:
        if not known:
            return f"One person{where}; identifying the activity…"
        label = known[0]
        subject = "A person"
        if posture in _POSTURE_ADJ and label not in _POSTURE:
            subject = f"A {_POSTURE_ADJ[posture]} person"
        return f"{subject} is {_phrase(label, details)}{where}{_duration(duration_s)}."

    # Several people: group identical activities, specific ones first.
    groups: dict[str, int] = {}
    for l in known:
        groups[l] = groups.get(l, 0) + 1
    if len(groups) == 1 and len(known) == n:
        only = next(iter(groups))
        return f"{_count(n).capitalize()} people are {_phrase(only, details)}{where}."
    # A posture shared by the group goes in the head ("Five people seated in
    # an office"), and is then not repeated as a clause.
    adj = _POSTURE_ADJ.get(group_posture or "")
    if adj:
        groups.pop(group_posture, None)
    order = sorted(groups, key=lambda l: (l in _POSTURE, -groups[l], l))
    clauses = []
    for l in order:
        k = groups[l]
        verb = "is" if k == 1 else "are"
        clauses.append(f"{_count(k)} {verb} {_phrase(l, details)}")
    # People whose activity is not confirmed are counted in the head but not
    # narrated: "two not yet identified" on every classroom title was noise.
    head = f"{_count(n).capitalize()} people{' ' + adj if adj else ''}{where}"
    if not clauses:
        return f"{head}."
    return f"{head}: " + ", ".join(clauses) + "."


class CaptionComposer:
    """Applies the window and hold rules above over time."""

    def __init__(self) -> None:
        self._frames: deque[tuple[float, int, Counter]] = deque()
        self._recent: deque[tuple[float, int, Counter]] = deque()
        # (time, Counter of (label, object), Counter of observed postures)
        self._context: deque[tuple[float, Counter, Counter, int]] = deque()
        self._main: tuple[str, float, float] | None = None   # label, since, last seen
        self._kept_details: dict[str, str] = {}
        self._kept_posture: str | None = None
        self._posture_frames: deque[tuple[float, Counter]] = deque()
        self._shown = _STARTING
        self._shown_key: tuple = ()
        self._pending: str | None = None
        self._pending_since = 0.0

    def reset(self) -> None:
        self._frames.clear()
        self._recent.clear()
        self._context.clear()
        self._main = None
        self._kept_details = {}
        self._kept_posture = None
        self._posture_frames.clear()
        self._shown = _STARTING
        self._shown_key = ()
        self._pending = None

    @property
    def text(self) -> str:
        return self._shown

    def update(self, annotations: list[dict], scene: str | None,
               now: float | None = None) -> str:
        now = time.monotonic() if now is None else now
        labels = Counter(
            a.get("label") for a in annotations
            if a.get("label") in _PHRASES
            and float(a.get("confidence", 0.0)) >= _MIN_CONFIDENCE
        )
        self._frames.append((now, len(annotations), labels))
        while self._frames and now - self._frames[0][0] > _WINDOW_S:
            self._frames.popleft()
        objects = Counter(
            (a.get("label"), a.get("detail")) for a in annotations if a.get("detail"))
        postures = Counter(a.get("posture") for a in annotations if a.get("posture"))
        self._context.append((now, objects, postures, len(annotations)))
        while self._context and now - self._context[0][0] > _CONTEXT_WINDOW_S:
            self._context.popleft()

        people = _median([n for _, n, _ in self._frames])
        recent = [c for t, _, c in self._frames if now - t <= _LABEL_WINDOW_S]
        counts = Counter()
        for label in {l for c in recent for l in c}:
            k = _median([c.get(label, 0) for c in recent])
            if k:
                counts[label] = k

        # Sticky recent highs (see _PEOPLE_STICK_S). Only once a group has
        # formed: a single person's caption must follow them immediately.
        self._recent.append((now, people, counts))
        while self._recent and now - self._recent[0][0] > _PEOPLE_STICK_S:
            self._recent.popleft()
        self._posture_frames.append((now, Counter({l: k for l, k in labels.items() if l in _POSTURE})))
        while self._posture_frames and now - self._posture_frames[0][0] > _GROUP_POSTURE_WINDOW_S:
            self._posture_frames.popleft()
        if people >= 2:
            people = max(n for _, n, _ in self._recent)
            # Posture/motion clauses in a group come from the longer window.
            slow = Counter()
            for label in _POSTURE:
                k = _median([c.get(label, 0) for _, c in self._posture_frames])
                if k:
                    slow[label] = k
            counts = Counter({l: k for l, k in counts.items() if l not in _POSTURE})
            counts.update({l: k for l, k in slow.items()
                           if k >= _GROUP_POSTURE_SHARE * people})
        groups: list[str] = []
        for label in sorted(counts):
            groups += [label] * counts[label]
        groups = groups[:people]

        # Corroborated context, each the majority reading of the last few s,
        # kept once said for as long as its activity continues.
        frames_n = len(self._context)
        details: dict[str, str] = {
            l: o for l, o in self._kept_details.items() if l in groups}
        for label in set(groups):
            seen = Counter()
            for _, objs, _, _ in self._context:
                for (l, obj), k in objs.items():
                    if l == label:
                        seen[obj] += 1
            if seen:
                obj, k = seen.most_common(1)[0]
                if k >= _CONTEXT_SHARE * frames_n:
                    details[label] = obj
        self._kept_details = details
        posture = group_posture = None
        if self._context:
            if people == 1:
                tally = Counter()
                for _, _, ps, _ in self._context:
                    tally.update(ps)
                if tally:
                    pst, k = tally.most_common(1)[0]
                    if k >= _POSTURE_SHARE * frames_n:
                        self._kept_posture = pst
                posture = self._kept_posture
            elif people >= 2:
                # Shared by the group: at least 80% of the people, on average.
                share = Counter()
                for _, _, ps, n in self._context:
                    for pst, k in ps.items():
                        share[pst] += k / max(n, 1)
                if share:
                    pst, total = share.most_common(1)[0]
                    if total / frames_n >= 0.8:
                        group_posture = pst

        # How long the lone person's main activity has been going on.
        duration = None
        if people != 1:
            self._kept_posture = None
        if people == 1 and groups and groups[0] not in _POSTURE:
            label = groups[0]
            if self._main and self._main[0] == label and now - self._main[2] <= _DURATION_GAP_S:
                self._main = (label, self._main[1], now)
            else:
                self._main = (label, now, now)
            duration = now - self._main[1]
        sentence = compose_sentence(groups + [None] * (people - len(groups)), scene,
                                    details=details, posture=posture,
                                    group_posture=group_posture, duration_s=duration)
        key = (people, tuple(sorted(groups)), scene)

        if sentence == self._shown:
            self._pending = None
        elif sentence != self._pending:
            self._pending, self._pending_since = sentence, now
            self._pending_key = key
        if any(g in _FALL for g in groups):
            hold = 0.0                  # an alert goes up at once
        elif _only_adds(self._shown_key, key) or _sharpens(self._shown_key, key):
            hold = _FAST_HOLD_S
        elif _only_count_differs(self._shown_key, key):
            hold = _COUNT_HOLD_S
        else:
            hold = _CAPTION_HOLD_S
        if self._pending is not None and now - self._pending_since >= hold:
            self._shown, self._shown_key, self._pending = self._pending, self._pending_key, None
        return self._shown


def _sharpens(old: tuple, new: tuple) -> bool:
    """True when `new` only replaces plain postures with specific activities
    ("sitting" -> "taking tablets"): more information, not a contradiction,
    so it goes up as fast as an addition. Measured on the Jetson, the full
    hold kept "A person is sitting" under a box already reading "taking
    tablets 73%" for over a second."""
    if not old:
        return False
    (n0, g0, s0), (n1, g1, s1) = old, new
    if n0 != n1 or (s0 and s0 != s1):
        return False
    gone = list(g0)
    for g in g1:
        if g in gone:
            gone.remove(g)
    added = list(g1)
    for g in g0:
        if g in added:
            added.remove(g)
    return bool(added) and all(g in _POSTURE for g in gone) \
        and all(g not in _POSTURE for g in added)


def _only_count_differs(old: tuple, new: tuple) -> bool:
    return bool(old) and old[0] != new[0] and old[1:] == new[1:]


def _only_adds(old: tuple, new: tuple) -> bool:
    """True when `new` keeps everything `old` said and only adds to it."""
    if not old:                       # nothing confirmed on screen yet
        return True
    (n0, g0, s0), (n1, g1, s1) = old, new
    if n0 != n1 or (s0 and s0 != s1):
        return False
    remaining = list(g1)
    for g in g0:
        if g not in remaining:
            return False
        remaining.remove(g)
    return True


def _median(values: list[int]) -> int:
    """Lower median: a tie between 1 and 2 people reads as 1, never 1.5."""
    if not values:
        return 0
    v = sorted(values)
    return v[(len(v) - 1) // 2]
