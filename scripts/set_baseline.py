"""Use a local recording as the left-hand side of the showcase comparison.

    python scripts/set_baseline.py dist/client-demo dist/cpu-baseline

The GPU showcase (kaggle/gpu_showcase.py) is the right-hand side. This takes
a recording of the same clips made elsewhere, by default this machine's CPU
with scripts/process_video.py, and attaches it as the left-hand side: its
videos go to <showcase>/before/ and showcase.json gets its summaries and the
side labels. Then run build_comparison.py and build_showcase.py.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path


def cpu_name() -> str:
    name = platform.processor() or "CPU"
    try:
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        name = out.splitlines()[0].strip() if out else name
    except Exception:
        pass
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("showcase", type=Path)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("--left-tag", default="WITHOUT GPU")
    ap.add_argument("--left-sub", default="laptop CPU")
    ap.add_argument("--compare", nargs="*",
                    help="clip names to show as side-by-side pairs (default: keep)")
    ap.add_argument("--same-models", action="store_true",
                    help="the CPU recording used the GPU's models and settings")
    args = ap.parse_args()

    data = json.loads((args.showcase / "showcase.json").read_text(encoding="utf-8"))
    expected = {Path(c["video"]).stem: c.get("expected") for c in data["clips"]}
    before_dir = args.showcase / "before"
    shutil.rmtree(before_dir, ignore_errors=True)
    before_dir.mkdir(parents=True)
    rows = []
    for f in sorted(args.baseline.glob("*_summary.json")):
        s = json.loads(f.read_text())
        stem = Path(s["video"]).stem
        if stem not in expected or not s.get("annotated_video"):
            continue
        video = Path(s["annotated_video"])
        if not video.exists():
            video = args.baseline / video.name
        shutil.copy(video, before_dir / video.name)
        s["annotated_video"] = video.name
        s["expected"] = expected[stem]
        rows.append(s)

    gpu = (data.get("gpu") or "GPU").replace("NVIDIA ", "")
    data["before"] = rows
    data["compare_left"] = {"tag": args.left_tag, "sub": args.left_sub}
    data["compare_right"] = {"tag": "WITH GPU", "sub": gpu}
    data["compare_title"] = f"Without a GPU vs with a GPU: same clips, same software"
    if args.same_models:
        data["compare_points"] = [
            f"<b>Identical software and models on both sides</b>: the VideoMAE activity "
            "model (87M parameters, a look every 0.4 s), the YOLOv8s pose detector, and "
            "the vision-language model that writes the detailed subtitles.",
            f"<b>Without GPU</b>: this laptop's CPU ({cpu_name()}) cannot run them in real "
            "time, so labels arrive late or not at all and subtitles stay basic.",
            f"<b>With GPU</b> ({gpu}): the same models keep up with the video.",
            "Same clips, 12 seconds each; every figure is what the screen showed.",
        ]
    else:
        data["compare_points"] = [
            f"<b>Without GPU</b>: this laptop's CPU ({cpu_name()}) runs the light models "
            "it can afford in real time: an 8M-parameter activity model that looks once "
            "every 1.5 s, and a small person detector at 416 px.",
            f"<b>With GPU</b> ({gpu}): a 10x larger activity model (VideoMAE) looking 2.5 "
            "times a second, a larger person detector, and 960 px detection for HD video.",
            "Same software, same clips, 12 seconds each; every figure is what the screen "
            "showed.",
        ]
    if args.compare:
        data["compare"] = [Path(c).stem for c in args.compare]
    (args.showcase / "showcase.json").write_text(json.dumps(data, indent=2))
    print(f"  attached {len(rows)} baseline clips from {args.baseline} as '{args.left_tag}'")


if __name__ == "__main__":
    main()
