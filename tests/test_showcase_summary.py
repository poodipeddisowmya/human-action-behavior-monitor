"""The comparison tab's summary with an extra device (scripts/add_showcase_side.py)."""
from src.utils.showcase import summarize


def _row(video, shown, first=None, fps=25.0, expected=None):
    r = {"video": video, "annotated_video": video.split(".")[0] + "_annotated.mp4",
         "labels_shown": shown, "first_seen_s": first or {}, "processing_fps": fps,
         "tracked_mean": 1.0}
    if expected:
        r["expected"] = expected
    return r


def _data(extra_clips):
    return {
        "before": [_row("a.mp4", {"standing": 1.0}, fps=10),
                   _row("b.mp4", {"standing": 1.0}, fps=10)],
        "clips": [_row("a.mp4", {"cooking": 0.9}, {"cooking": 1.0}, expected="cooking"),
                  _row("b.mp4", {"eating": 0.3}, {"eating": 2.0}, expected="eating")],
        "extra_sides": [{"key": "jetson", "tag": "EDGE GPU", "sub": "Jetson",
                         "short": "Jetson", "clips": extra_clips}],
    }


def test_extra_side_totals_and_clips():
    # The extra side's rows carry no "expected": it comes from the GPU side.
    s = summarize(_data([_row("a.mp4", {"cooking": 0.6}, {"cooking": 0.5}, fps=22),
                         _row("b.mp4", {"eating": 0.7}, {"eating": 1.5}, fps=22)]))
    (x,) = s["extra"]
    assert (x["key"], x["short"]) == ("jetson", "Jetson")
    assert x["totals"]["clear"] == 2 and s["totals"]["right"]["clear"] == 1
    assert x["totals"]["median_fps"] == 22
    a = next(c for c in s["clips"] if c["name"] == "a")
    assert a["extra"]["jetson"]["share"] == 0.6
    assert a["extra"]["jetson"]["first_s"] == 0.5


def test_clip_missing_on_extra_side_is_null():
    s = summarize(_data([_row("a.mp4", {"cooking": 0.6}, {"cooking": 0.5})]))
    b = next(c for c in s["clips"] if c["name"] == "b")
    assert b["extra"]["jetson"] is None
    assert s["extra"][0]["totals"]["clear"] == 1


def test_no_extra_sides_keeps_two_sided_shape():
    d = _data([])
    del d["extra_sides"]
    s = summarize(d)
    assert s["extra"] == []
    assert all(c["extra"] == {} for c in s["clips"])
