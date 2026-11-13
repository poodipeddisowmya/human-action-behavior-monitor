"""Measure every candidate improvement on the same clips, on a Kaggle T4.

One run, one table: for each configuration, every clip is played through the
real pipeline (scripts/process_video.py, 12 s each) and scored on what the
viewer would see:
    clear       correct activity on screen >= 50% of the time
    share       fraction of on-screen labels that are the correct one
    first_s     seconds until the correct label first appears
    flicker     on-screen label changes per 10 s
    tracked     people tracked per frame (crowd clips; detection recall)
Output: /kaggle/working/benchmark.json (+ per-config folders with videos)
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/kaggle/working")
APP = Path("/root/app")
PY = "/root/ham/venv/bin/python"
SECONDS = 12
CROWD = ["office-people.mp4", "store-aisle.mp4", "street-surveillance.avi"]
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}


def sh(cmd, **kw):
    return subprocess.run(cmd, cwd=APP, env={**ENV, **kw.pop("env", {})},
                          capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)


def expected_labels() -> dict[str, str]:
    out = sh([PY, "-c", "import sys,json;sys.path.insert(0,'scripts');"
                        "from eval_offline import EXPECTED;print(json.dumps(EXPECTED))"]).stdout
    exp = {f"data/samples/{k}": v for k, v in json.loads(out.strip().splitlines()[-1]).items()}
    for f in sorted((APP / "data" / "samples").glob("hd-*.mp4")):
        exp[f"data/samples/{f.name}"] = f.stem.split("-")[1]          # "hd-eating-2"
    for f in sorted((APP / "data" / "candidates" / "720p").glob("*.mp4")):
        exp[f"data/candidates/720p/{f.name}"] = f.name.split("-")[0]   # "drinking-123.mp4"
    return exp


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def run_config(name: str, env: dict, clips: list[str], expected: dict[str, str]) -> dict:
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = sh([PY, "scripts/process_video.py", *clips, "--out", str(out),
            "--min-seconds", str(SECONDS), "--max-width", "960"], env=env)
    (out / "run.log").write_text(r.stdout[-200000:])
    rows = []
    for c in clips:
        f = out / f"{Path(c).stem}_summary.json"
        if not f.exists():
            rows.append({"clip": c, "error": "no summary"})
            continue
        s = json.loads(f.read_text())
        exp = expected.get(c)
        shown = s.get("labels_shown") or {}
        row = {"clip": c, "expected": exp, "fps": s.get("processing_fps"),
               "flicker": s.get("label_changes_per_10s"),
               "tracked_mean": s.get("tracked_mean"), "tracked_max": s.get("tracked_max"),
               "top": dict(list(shown.items())[:3])}
        if exp:
            row["share"] = shown.get(exp, 0.0)
            row["clear"] = row["share"] >= 0.5
            row["first_s"] = (s.get("first_seen_s") or {}).get(exp)
        rows.append(row)
    lab = [r for r in rows if r.get("expected")]
    firsts = [r["first_s"] for r in lab if r.get("first_s") is not None]
    agg = {
        "minutes": round((time.time() - t0) / 60, 1),
        "clear": f"{sum(r.get('clear', False) for r in lab)}/{len(lab)}",
        "mean_share": _mean([r.get("share", 0) for r in lab]),
        "median_first_s": _median(firsts),
        "never_correct": sum(r.get("first_s") is None for r in lab),
        "mean_flicker": _mean([r.get("flicker") for r in rows]),
        "crowd_tracked_mean": {Path(r["clip"]).stem: r["tracked_mean"] for r in rows
                               if Path(r["clip"]).name in CROWD},
        "median_fps": _median([r.get("fps") for r in rows]),
    }
    print(f"== {name}: {json.dumps(agg)}", flush=True)
    return {"env": env, "summary": agg, "clips": rows}


def main() -> None:
    if not APP.exists():
        src = Path(glob.glob("/kaggle/input/**/scripts/run_pipeline.py", recursive=True)[0])
        shutil.copytree(src.parent.parent, APP)
    # Fail loudly on stale project files rather than measure the wrong code.
    builds = sorted(p.name for p in APP.glob("BUILD-*.txt"))
    print("project build:", builds, flush=True)
    if not (APP / "scripts" / "startup_probe.py").exists():
        sys.exit("STALE DATASET: this run got an older project version; re-run kaggle_auto.py")
    subprocess.run([sys.executable, "scripts/kaggle_launch.py", "setup"], cwd=APP, env=ENV, check=True)
    # The stronger activity model, fetched once (the app itself reads offline).
    r = sh([PY, "-c", "from transformers import VideoMAEForVideoClassification as M, "
                      "VideoMAEImageProcessor as P;c='MCG-NJU/videomae-base-finetuned-kinetics';"
                      "M.from_pretrained(c);P.from_pretrained(c);print('videomae ready')"],
           env={"HF_HUB_OFFLINE": "0"})
    print(r.stdout[-300:], r.stderr[-500:], flush=True)

    expected = expected_labels()
    samples = [c for c in expected if c.startswith("data/samples/")]
    hd = [c for c in expected if c.startswith("data/candidates/")]
    crowd = [f"data/samples/{c}" for c in CROWD if (APP / "data" / "samples" / c).exists()]

    report = {"build": builds, "gpu": sh(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).stdout.strip(),
              "seconds_per_clip": SECONDS, "probe": [], "configs": {}}
    pr = sh([PY, "scripts/startup_probe.py", "data/samples/demo-cooking.mp4",
             "data/samples/demo-writing.avi"])
    report["probe"] = [json.loads(l[6:]) for l in pr.stdout.splitlines() if l.startswith("PROBE ")]
    print("probe:", report["probe"], pr.stderr[-800:] if not report["probe"] else "", flush=True)

    # The shipped GPU profile (VideoMAE, HD-only 960 px detector, flicker
    # rule), and the same code with the CPU's S3D model for comparison.
    plan = [
        ("final", {}, samples + crowd + hd),
        ("final_with_s3d", {"DEEP_BACKEND": "s3d"}, samples),
    ]
    for name, env, clips in plan:
        report["configs"][name] = run_config(name, env, clips, expected)
        (OUT / "benchmark.json").write_text(json.dumps(report, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
