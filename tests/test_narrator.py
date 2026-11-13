"""Detailed subtitles (src/description/narrator.py) and how they are shown.

The model itself is not loaded here (it is GPU-only in practice); these pin
the parts that decide what reaches the screen.
"""
from __future__ import annotations

import numpy as np

from src.description.describer import SceneDescriber
from src.description.narrator import _MAX_WORDS, _sentence, build_prompt


def test_sentence_is_tidied():
    assert _sentence('  "a person is eating a sandwich"\n') == "A person is eating a sandwich."


def test_numbered_people_become_plain_words():
    """Seen on a T4: 'Person1 is eating a sandwich ... person2 is holding a fork.'"""
    out = _sentence("Person1 is eating a sandwich with avocado. person2 is holding a fork.")
    assert out == "A person is eating a sandwich with avocado. Another person is holding a fork."
    assert "Person" not in _sentence("Two people dance. Person 3 watches.").split(". ")[1][:7]


def test_overlong_output_is_cut_at_a_full_stop():
    long = "A person swims in a pool. " + " ".join(["word"] * 60)
    out = _sentence(long)
    assert out == "A person swims in a pool."
    assert len(out.split()) <= _MAX_WORDS


def test_prompt_is_grounded_in_what_was_measured():
    p = build_prompt(1, "swimming", "in a swimming pool")
    assert "exactly one person in view" in p and "main activity is swimming" in p
    assert "the place is a swimming pool" in p
    assert "3 people in view" in build_prompt(3, None, None)
    assert "main activity" not in build_prompt(3, None, None)


def test_gender_is_never_stated():
    """Seen on a T4: a woman dancer described as 'a man'."""
    assert _sentence("A man in a white top is dancing.") == "A person in a white top is dancing."
    assert _sentence("A man and a woman sit, she is eating.") == "Two people sit, they are eating."
    assert _sentence("Two women walk; his bag.") == "Two people walk; their bag."
    assert _sentence("The manager waits.") == "The manager waits."     # whole words only


class _Narrator:
    available = True

    def __init__(self, text):
        self.text, self.submitted, self.was_reset = text, [], False

    def submit(self, frame, people, activity, place):
        self.submitted.append((people, activity, place))

    def reset(self):
        self.was_reset = True


ANN = [{"label": "dancing", "person_id": 1, "confidence": 0.9, "xyxy": (0, 0, 10, 10)},
       {"label": "standing", "person_id": 2, "confidence": 0.9, "xyxy": (0, 0, 10, 10)}]


def test_fresh_description_replaces_the_template_caption():
    n = _Narrator("Two people are dancing at a party under coloured lights.")
    d = SceneDescriber(scene=False, narrator=n)
    out = d.update(np.zeros((720, 1280, 3), np.uint8), ANN)
    assert out["text"] == "Two people are dancing at a party under coloured lights."
    assert n.submitted[-1] == (2, "dancing", None)      # activity, not posture


def test_low_resolution_video_keeps_the_template_caption():
    """Seen on a T4: a 320x240 clip narrated as 'using a blue shirt as a broom handle'."""
    n = _Narrator("A person uses a blue shirt as a broom handle.")
    d = SceneDescriber(scene=False, narrator=n)
    out = d.update(np.zeros((240, 320, 3), np.uint8), ANN)
    assert "broom handle" not in out["text"] and not n.submitted
    out = d.update(np.zeros((720, 1280, 3), np.uint8), ANN)
    assert "broom handle" in out["text"]


def test_without_a_description_the_template_caption_is_shown():
    d = SceneDescriber(scene=False, narrator=_Narrator(None))
    out = d.update(np.zeros((8, 8, 3), np.uint8), ANN)
    assert out["text"] and out["text"] != "None"


def test_unavailable_narrator_is_ignored_and_reset_propagates():
    n = _Narrator("x")
    n.available = False
    assert SceneDescriber(scene=False, narrator=n)._narrator is None
    n2 = _Narrator("x")
    d = SceneDescriber(scene=False, narrator=n2)
    d.reset()
    assert n2.was_reset
