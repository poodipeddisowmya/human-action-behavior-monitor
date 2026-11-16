"""Record the showcase clip set on THIS machine, exactly as the Kaggle T4 did.

    python scripts/record_showcase.py --out dist/jetson-showcase

Same clips (every labelled demo clip, the HD stock footage and the crowd
scenes), same 12 s per clip, same 1280 px output cap, same pipeline
(scripts/process_video.py) as kaggle/gpu_showcase.py, so the resulting
showcase.json is directly comparable with the T4 and laptop-CPU recordings
in data/showcase/showcase.json (scripts/compare_devices.py puts them side
by side). Kaggle-free: runs on a Jetson, a desktop GPU or a CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JETSON = Path("/etc/nv_tegra_release").exists()

SECONDS = 12
CROWD = ["office-people.mp4", "store-aisle.mp4", "street-surveillance.avi"]


def device_name() -> str:
    """'Jetson Orin Nano ...', the GPU's name, or the CPU's."""
    model = Path("/proc/device-tree/model")
    if model.exists():
        return model.read_text(errors="ignore").strip("\x00 \n")
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return platform.processor() or platform.machine()


def power_mode() -> str | None:
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True,
                             timeout=10).stdout
        return out.splitlines()[0].split(":", 1)[1].strip()
    except Exception:
        return None


def expected_labels() -> dict[str, str | None]:
    """eval_offline.EXPECTED, read in a child process: importing it here would
    load the app (and a CUDA context) into this process, which only waits."""
    out = subprocess.run(
        [sys.executable, "-c", "import sys,json;sys.path.insert(0,'scripts');"
                               "from eval_offline import EXPECTED;print(json.dumps(EXPECTED))"],
        cwd=ROOT, capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "DEVICE": "cpu"}).stdout
    return json.loads(out.strip().splitlines()[-1])


def reclaim_page_cache(gb: float = 2.0) -> None:
    """Jetson only: make room for the models in the shared CPU/GPU memory.

    CUDA allocations there do not evict the page cache, and every clip reads
    ~1.5 GB of weights into it; with the cache full the next clip's models
    fail to load ("NvMapMemAlloc error 12"). Touching a large buffer once
    makes the kernel evict it, without the sudo that drop_caches needs.
    """
    subprocess.run([sys.executable, "-c",
                    f"import numpy as np; np.ones(int({gb}*2**30)//8).sum()"], check=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "device-showcase")
    ap.add_argument("--clips", nargs="*", help="subset of clip file names")
    a = ap.parse_args()

    expected = expected_labels()
    for f in sorted((ROOT / "data" / "samples").glob("hd-*.mp4")):
        expected[f.name] = f.stem.split("-")[1]
    clips = [c for c in [*expected, *CROWD] if (ROOT / "data" / "samples" / c).exists()]
    if a.clips:
        clips = [c for c in clips if c in a.clips]

    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    degraded: dict[str, list[str]] = {}
    for i, c in enumerate(clips, 1):
        if JETSON:
            reclaim_page_cache()
        log = a.out / f"{Path(c).stem}.log"
        with log.open("w", encoding="utf-8") as fh:
            subprocess.run([sys.executable, "scripts/process_video.py", f"data/samples/{c}",
                            "--out", str(a.out), "--min-seconds", str(SECONDS),
                            "--max-width", "1280"],
                           cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                           stdout=fh, stderr=subprocess.STDOUT, check=False)
        text = log.read_text(encoding="utf-8", errors="ignore")
        lost = [name for name, marker in (("detailed captions (SmolVLM2)", "scene narrator unavailable"),
                                          ("posture check (CLIP)", "posture check unavailable"),
                                          ("place names (CLIP)", "scene description unavailable"),
                                          ("VideoMAE (fell back to S3D)", "VideoMAE unavailable"),
                                          ("activity model", "[deep] unavailable"))
                if marker in text]
        # Failures while running, not just at load: on a Jetson a model can
        # load and then run out of memory on individual looks.
        for name, marker in (("activity looks failed", "[deep] inference error"),
                             ("subtitle sentences failed", "scene narration failed"),
                             ("scene looks failed", "scene look failed"),
                             ("detection failures", "detection failed")):
            n = text.count(marker)
            if n:
                lost.append(f"{n} {name}")
        if lost:
            degraded[c] = lost
        ok = (a.out / f"{Path(c).stem}_summary.json").exists()
        print(f"[{i}/{len(clips)}] {c}: {'ok' if ok else 'FAILED'}"
              + (f"  (without: {', '.join(lost)})" if lost else ""), flush=True)

    rows = []
    for c in clips:
        f = a.out / f"{Path(c).stem}_summary.json"
        if f.exists():
            s = json.loads(f.read_text())
            s["expected"] = expected.get(c)
            s["annotated_video"] = (Path(s["annotated_video"]).name
                                    if s.get("annotated_video") else None)
            rows.append(s)

    sys.path.insert(0, str(ROOT))
    from config import settings  # after the recording: it initialises CUDA
    (a.out / "showcase.json").write_text(json.dumps({
        "gpu": device_name(),
        "power_mode": power_mode(),
        "recorded": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "seconds_per_clip": SECONDS, "minutes": round((time.time() - t0) / 60, 1),
        "settings": {"device": settings.device,
                     "deep_backend": settings.model.deep_backend,
                     "detector": settings.model.detector_weights,
                     "detector_imgsz": settings.model.detector_imgsz,
                     "deep_min_interval_s": settings.model.deep_min_interval_s,
                     "vlm_captions": settings.model.vlm_captions},
        # Optional models that failed to load on this device (memory), per clip.
        "degraded": degraded,
        "clips": rows}, indent=2))
    print(f"recorded {len(rows)}/{len(clips)} clips in {(time.time() - t0) / 60:.1f} min "
          f"-> {a.out / 'showcase.json'}")


if __name__ == "__main__":
    main()
