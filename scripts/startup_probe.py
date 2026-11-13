"""How long after start does the first activity label appear, and where does
the time go?

    python scripts/startup_probe.py data/samples/demo-cooking.mp4

Starts the real pipeline on a looping clip and timestamps each stage of the
first-label path: first tracking result -> first clip handed to the activity
model -> first activity-model answer -> first activity label on screen.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.input import FileSource  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402

POSTURES = {"standing", "sitting", "walking", "running", "detecting…"}


def probe(clip: Path, seconds: float = 10.0) -> dict:
    marks: dict[str, float] = {}
    t0 = [0.0]

    def mark(key: str) -> None:
        if t0[0] and key not in marks:
            marks[key] = round(time.time() - t0[0], 2)

    t_build = time.time()
    p = Pipeline(FileSource(clip, loop=True), on_frame=lambda *a: mark("first_frame"))
    build_s = round(time.time() - t_build, 2)
    trk, clf = p._tracker, p._classifier

    upd = trk.update
    trk.update = lambda img: (lambda r: (mark("first_tracking" if r else "first_tracker_call"), r)[1])(upd(img))
    cupd = clf.update

    def classify(*a, **k):
        r = cupd(*a, **k)
        if r is not None and r.label not in POSTURES:
            mark("first_activity_label")
        return r
    clf.update = classify
    if hasattr(clf, "_submit"):
        sub = clf._submit
        clf._submit = lambda *a, **k: (mark("first_submit"), sub(*a, **k))[1]
        inf = clf._infer
        clf._infer = lambda *a, **k: (lambda r: (mark("first_model_answer"), r)[1])(inf(*a, **k))

    t0[0] = time.time()
    p.start()
    time.sleep(seconds)
    p.stop()
    return {"clip": clip.name, "build_s": build_s, **marks}


if __name__ == "__main__":
    for c in sys.argv[1:] or ["data/samples/demo-cooking.mp4"]:
        print("PROBE " + json.dumps(probe(Path(c))), flush=True)
