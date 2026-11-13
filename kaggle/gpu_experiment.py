"""Which GPU-only setting changes the labels? A controlled experiment on a T4.

Runs the real-time accuracy harness on the clips that differ between CPU and
GPU, once per configuration, switching ONE GPU setting back to its CPU value
at a time. Output: /kaggle/working/experiment.json
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path("/kaggle/working")
APP = Path("/root/app")
PY = "/root/ham/venv/bin/python"
CLIPS = ["demo-smoking.mp4", "demo-cycling.avi", "demo-exercising-pushup.avi",
         "demo-exercising-squats.avi", "demo-basic-sit.avi", "demo-drinking.mp4"]
CONFIGS = {
    "gpu_shipped":       {},
    "imgsz_416":         {"DETECTOR_IMGSZ": "416"},
    "fp32_detector":     {"DETECTOR_HALF": "0"},
    "deep_every_1.5s":   {"DEEP_MIN_INTERVAL_S": "1.5"},
    "nano_detector":     {"DETECTOR_WEIGHTS": "yolov8n-pose.pt"},
    "cpu_profile_on_gpu": {"DETECTOR_IMGSZ": "416", "DETECTOR_HALF": "0",
                           "DEEP_MIN_INTERVAL_S": "1.5", "DETECTOR_WEIGHTS": "yolov8n-pose.pt"},
}


def main() -> None:
    if not APP.exists():
        src = Path(glob.glob("/kaggle/input/**/scripts/run_pipeline.py", recursive=True)[0])
        shutil.copytree(src.parent.parent, APP)
    subprocess.run([sys.executable, "scripts/kaggle_launch.py", "setup"], cwd=APP, check=True)
    results = {}
    for name, env in CONFIGS.items():
        out = subprocess.run([PY, "scripts/eval_offline.py", "--realtime", "--clips", *CLIPS],
                             cwd=APP, env={**os.environ, "PYTHONIOENCODING": "utf-8", **env},
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace").stdout
        (OUT / f"exp_{name}.log").write_text(out)
        rows = re.findall(r"^\s+(\S+)\s+.+?\s{2,}(.+?)\s+(PASS|FAIL)\s+[\d.]+$", out, re.M)
        results[name] = {"passed": sum(r[2] == "PASS" for r in rows), "total": len(rows),
                         "clips": {r[0]: f"{r[2]} ({r[1].strip()})" for r in rows}}
        print(name, json.dumps(results[name]), flush=True)
        (OUT / "experiment.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
