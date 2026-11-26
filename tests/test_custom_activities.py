"""Activities trained on your own clips (custom_activities.py and
scripts/train_activities.py), on synthetic features: no video or GPU."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from src.action_recognition import custom_activities as ca
from src.action_recognition.custom_activities import OTHER, ActivityHead, CustomActivities, train_head

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_activities", ROOT / "scripts" / "train_activities.py")
ta = importlib.util.module_from_spec(spec)
sys.modules["train_activities"] = ta
spec.loader.exec_module(ta)


def _data(n=60, dim=32, seed=0):
    """Two separable activities plus 'other', in feature space."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 3, (3, dim))
    x = np.concatenate([centres[c] + rng.normal(0, 1, (n, dim)) for c in range(3)]).astype(np.float32)
    y = np.repeat(np.arange(3), n)
    return x, y, centres


def test_trains_and_separates():
    x, y, centres = _data()
    head = train_head(x, y, ["fighting", "sleeping at desk", OTHER], "s3d")
    assert (head.probs(x).argmax(1) == y).mean() > 0.95
    assert head.activities == ["fighting", "sleeping at desk"]
    assert head.probs(centres[1]).argmax() == 1          # a single look


def test_unbalanced_classes_still_learned():
    """A rare activity counts as much as plentiful 'other' (class weights)."""
    x, y, centres = _data(n=200)
    keep = (y != 0) | (np.arange(len(y)) % 20 == 0)      # only 10 looks of class 0
    head = train_head(x[keep], y[keep], ["fighting", "sleeping at desk", OTHER], "s3d")
    assert head.probs(centres[0]).argmax() == 0


def test_save_load_round_trip_and_no_pickle(tmp_path):
    x, y, _ = _data()
    head = train_head(x, y, ["a b", "c", OTHER], "videomae")
    head.threshold = 0.7
    head.meta = {"phrases": {"a b": "doing a b"}}
    head.save(tmp_path / "m.npz", tmp_path / "m.json")
    back = ActivityHead.load(tmp_path / "m.npz", tmp_path / "m.json")
    assert back.classes == head.classes and back.threshold == 0.7 and back.backend == "videomae"
    np.testing.assert_allclose(back.probs(x[:5]), head.probs(x[:5]), rtol=1e-5)
    assert back.meta["phrases"] == {"a b": "doing a b"}


def test_rejects_mismatched_file(tmp_path):
    x, y, _ = _data()
    head = train_head(x, y, ["a", "b", OTHER], "s3d")
    head.save(tmp_path / "m.npz", tmp_path / "m.json")
    import json
    meta = json.loads((tmp_path / "m.json").read_text())
    meta["classes"] = ["a", OTHER]                        # two names, three rows of weights
    (tmp_path / "m.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        ActivityHead.load(tmp_path / "m.npz", tmp_path / "m.json")


def _live(threshold=0.6):
    x, y, centres = _data()
    head = train_head(x, y, ["fighting", "sleeping at desk", OTHER], "s3d")
    head.threshold = threshold
    return CustomActivities(head), centres


def test_live_needs_three_of_four_looks(monkeypatch):
    live, centres = _live()
    live.strong = 1.01                                    # disable the one-strong-look path
    live.observe(7, centres[1])
    live.observe(7, centres[2])                           # an ordinary look in between
    live.observe(7, centres[1])
    assert live.current(7) is None                        # two of three is not enough
    live.observe(7, centres[1])
    assert live.current(7).label == "sleeping at desk"    # three of the last four
    assert live.current(8) is None                        # another person: nothing


def test_live_other_never_shown_and_hold_expires(monkeypatch):
    live, centres = _live()
    for _ in range(3):
        live.observe(1, centres[2])
    assert live.current(1) is None                        # "other" is never a label
    for _ in range(3):
        live.observe(2, centres[0])
    assert live.current(2).label == "fighting"
    t = ca.time.monotonic()
    monkeypatch.setattr(ca.time, "monotonic", lambda: t + live.hold_s + 1)
    assert live.current(2) is None                        # unsupported for too long
    live.reset()
    assert live.current(2) is None


def test_load_for_missing_or_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(ca.settings, "models_dir", tmp_path)
    assert CustomActivities.load_for("s3d") is None       # nothing trained
    x, y, _ = _data()
    head = train_head(x, y, ["a", "b", OTHER], "s3d")
    head.save(*ca.model_paths("s3d"))
    assert CustomActivities.load_for("s3d").activities == ["a", "b"]
    assert CustomActivities.load_for("videomae") is None  # trained for the other backbone
    monkeypatch.setattr(ca.settings, "custom_activities", {"enabled": False})
    assert CustomActivities.load_for("s3d") is None


def test_live_rule_matches_monitor():
    p = np.array([[0.7, 0.3], [0.2, 0.8], [0.7, 0.3]])
    assert ta.live_rule_fires(p, 0, 0.6, (2, 3, 0.9))     # looks 1 and 3 of the last 3
    assert not ta.live_rule_fires(np.array([[0.7, 0.3], [0.2, 0.8], [0.2, 0.8], [0.7, 0.3]]), 0, 0.6, (2, 3, 0.9))
    assert ta.live_rule_fires(np.array([[0.95, 0.05]]), 0, 0.6, (2, 3, 0.9))   # one strong look
    assert not ta.live_rule_fires(np.array([[0.95, 0.05]]), 0, 0.94, (2, 3, 0.9))  # not past a strict threshold


def test_threshold_choice_prefers_no_false_alarms():
    table = {t: {"false_alarm_clips": 2 if t < 0.7 else 0} for t in ta.THRESHOLDS}
    assert ta.choose_threshold(table) == 0.7
    table = {t: {"false_alarm_clips": 3 if t < 0.9 else 1} for t in ta.THRESHOLDS}
    assert ta.choose_threshold(table) == 0.9              # fewest, when none is clean


def test_discover_names_and_problems(tmp_path):
    (tmp_path / "Sleeping_At  Desk").mkdir()
    (tmp_path / "Sleeping_At  Desk" / "a.mp4").write_bytes(b"x")
    (tmp_path / "Sleeping_At  Desk" / "phrase.txt").write_text("asleep at their desk\n")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "b.mp4").write_bytes(b"x")
    (tmp_path / "!").mkdir()                             # not a usable activity name
    classes, other, phrases, problems = ta.discover(tmp_path, [])
    assert list(classes) == ["sleeping at desk"]
    assert phrases == {"sleeping at desk": "asleep at their desk"}
    assert [p.name for p in other] == ["b.mp4"]
    assert any("'!'" in p for p in problems)
    assert ta.probe(tmp_path / "other" / "b.mp4")       # not a video: a reason is given
