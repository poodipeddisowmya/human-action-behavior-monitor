"""On-screen label stability (src/temporal/stabilizer.py).

Measured on a T4: on low-resolution clips the posture label flipped between
running / walking / sitting every 0.1-0.5 s, because the smoother's 3-update
window is ~0.1 s at GPU frame rates.
"""
from __future__ import annotations

from src.temporal.smoother import SmoothedPrediction as P
from src.temporal.stabilizer import LabelStabilizer


def _run(stab, seq, dt=0.04):
    shown = []
    for i, label in enumerate(seq):
        shown.append(stab.update(1, P(label, 0.7), i * dt).label)
    return shown


def test_brief_posture_wobble_never_reaches_the_screen():
    seq = ["walking"] * 10 + ["running"] * 5 + ["walking"] * 10    # 0.2 s of "running"
    assert set(_run(LabelStabilizer(0.8), seq)) == {"walking"}


def test_lasting_posture_change_is_shown_within_the_window():
    seq = ["standing"] * 30 + ["sitting"] * 40                    # sitting for 1.6 s
    shown = _run(LabelStabilizer(0.8), seq)
    assert shown[-1] == "sitting"
    delay = (shown.index("sitting") - 30) * 0.04
    assert 0.2 <= delay <= 0.8, delay        # never instant, never stale


def test_alternating_raw_labels_show_the_dominant_one_not_the_first():
    """The first rule (unbroken persistence) stuck on whatever came first:
    a walking clip whose raw label alternated showed "running" 81% of the
    time on a T4."""
    pattern = ["walking", "walking", "running", "walking", "sitting"]  # 60% walking
    seq = ["running"] * 3 + pattern * 20
    shown = _run(LabelStabilizer(0.8), seq)
    assert shown[-1] == "walking"
    assert shown[len(shown) // 2:].count("walking") / (len(shown) - len(shown) // 2) > 0.9


def test_activity_replaces_posture_immediately():
    shown = _run(LabelStabilizer(0.8), ["standing"] * 5 + ["cooking"])
    assert shown[-1] == "cooking"


def test_fall_is_never_delayed():
    shown = _run(LabelStabilizer(0.8), ["walking"] * 5 + ["falling"])
    assert shown[-1] == "falling"


def test_placeholder_gives_way_at_once():
    shown = _run(LabelStabilizer(0.8), ["detecting…"] * 3 + ["sitting"])
    assert shown[-1] == "sitting"


def test_zero_dwell_disables_it():
    seq = ["walking", "running", "walking"]
    assert _run(LabelStabilizer(0), seq) == seq


def test_people_are_independent_and_reset_works():
    s = LabelStabilizer(0.8)
    s.update(1, P("walking", 0.7), 0.0)
    s.update(2, P("sitting", 0.7), 0.0)
    assert s.update(2, P("sitting", 0.7), 0.1).label == "sitting"
    s.reset(1)
    assert s.update(1, P("running", 0.7), 0.2).label == "running"   # fresh start
