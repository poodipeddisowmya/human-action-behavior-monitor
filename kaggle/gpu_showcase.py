"""Record the client showcase on a Kaggle GPU (a "script" kernel).

Every labelled demo clip plus the crowded scenes, run through the exact live
pipeline on the T4 and saved as annotated H.264 videos with their event logs.
Short clips loop to SECONDS, as they do in the live dashboard, so the labels
have time to settle. Output (downloaded by scripts/kaggle_auto.py showcase):
    /kaggle/working/showcase/<clip>_annotated.mp4, _summary.json, _events.json
    /kaggle/working/showcase/showcase.json
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/kaggle/working/showcase")
APP = Path("/root/app")
PY = "/root/ham/venv/bin/python"
SECONDS = 12
CROWD = ["office-people.mp4", "store-aisle.mp4", "street-surveillance.avi"]
# The comparison's left side can be recorded elsewhere (scripts/set_baseline.py,
# e.g. a laptop CPU); then the previous-build pass here is not needed.
RECORD_BEFORE = False
# The previous build: S3D activity model, no on-screen label stabilisation,
# one detector size for all video.
BEFORE_ENV = {"DEEP_BACKEND": "s3d", "LABEL_MIN_DWELL_S": "0", "DETECTOR_IMGSZ_HD": "0"}
# Clips recorded under both builds for the side-by-side section.
# Chosen from the T4 benchmark (2026-11-05) among clips whose result held
# across runs; the low-res demo-smoking clip read "drinking" on one run, so
# it is not used. The page also shows totals over ALL demo clips.
COMPARE = ["hd-swimming.mp4", "hd-dancing-3.mp4", "hd-cooking.mp4", "hd-eating.mp4",
           "hd-drinking.mp4", "hd-smoking-2.mp4", "demo-cleaning-mopping.avi",
           "demo-using-computer.avi", "demo-writing.avi", "street-surveillance.avi"]


def main() -> None:
    os.environ["MPLBACKEND"] = "Agg"
    if not APP.exists():
        src = Path(glob.glob("/kaggle/input/**/scripts/run_pipeline.py", recursive=True)[0])
        shutil.copytree(src.parent.parent, APP)
    subprocess.run([sys.executable, "scripts/kaggle_launch.py", "setup"], cwd=APP, check=True)

    # The labelled clip list lives in eval_offline.py, whose imports need the
    # app's environment (ultralytics etc.), not this kernel's Python.
    EXPECTED = json.loads(subprocess.run(
        [PY, "-c", "import sys,json;sys.path.insert(0,'scripts');"
                   "from eval_offline import EXPECTED;print(json.dumps(EXPECTED))"],
        cwd=APP, capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}).stdout.strip().splitlines()[-1])

    # HD stock footage added to the library (expected label is in the name).
    for f in sorted((APP / "data" / "samples").glob("hd-*.mp4")):
        EXPECTED[f.name] = f.stem.split("-")[1]
    # Activities added after Kinetics (making tea, taking tablets, falling
    # down), on third-party clips in data/new-activities (same list as
    # scripts/record_showcase.py, which the Jetson records with).
    sys.path.insert(0, str(APP / "scripts"))
    from record_showcase import NEW_ACTIVITY_CLIPS
    EXPECTED.update(NEW_ACTIVITY_CLIPS)

    def path(c: str) -> str:
        return f"data/new-activities/{c}" if c in NEW_ACTIVITY_CLIPS else f"data/samples/{c}"

    clips = [c for c in [*EXPECTED, *CROWD] if (APP / path(c)).exists()]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    subprocess.run([PY, "scripts/process_video.py",
                    *[path(c) for c in clips],
                    "--out", str(OUT), "--min-seconds", str(SECONDS), "--max-width", "1280"],
                   cwd=APP, env={**os.environ, "PYTHONIOENCODING": "utf-8"}, check=False)

    # BEFORE: the same clips under the previous build's settings, on the same
    # GPU, for side-by-side comparison (scripts/build_comparison.py).
    # Every labelled clip (for the totals table) plus the comparison clips.
    compare = [c for c in COMPARE if (APP / "data" / "samples" / c).exists()]
    before_clips = list(dict.fromkeys([c for c in clips if EXPECTED.get(c)] + compare))
    if not RECORD_BEFORE:
        before_clips = []
    if before_clips:
        subprocess.run([PY, "scripts/process_video.py",
                        *[f"data/samples/{c}" for c in before_clips],
                        "--out", str(OUT / "before"), "--min-seconds", str(SECONDS),
                        "--max-width", "1280"],
                       cwd=APP, env={**os.environ, "PYTHONIOENCODING": "utf-8", **BEFORE_ENV},
                       check=False)

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()

    def rows_for(folder: Path, names: list[str]) -> list[dict]:
        rows = []
        for c in names:
            f = folder / f"{Path(c).stem}_summary.json"
            if f.exists():
                s = json.loads(f.read_text())
                s["expected"] = EXPECTED.get(c)
                s["annotated_video"] = (Path(s["annotated_video"]).name
                                        if s.get("annotated_video") else None)
                rows.append(s)
        return rows

    rows = rows_for(OUT, clips)
    (OUT / "showcase.json").write_text(json.dumps({
        "gpu": gpu[0] if gpu else None,
        "recorded": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "seconds_per_clip": SECONDS, "minutes": round((time.time() - t0) / 60, 1),
        "before_settings": BEFORE_ENV,
        "clips": rows,
        "compare": [Path(c).stem for c in compare],
        "before": rows_for(OUT / "before", before_clips)}, indent=2))
    print(f"recorded {len(rows)}/{len(clips)} clips (+{len(before_clips)} before) "
          f"in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
