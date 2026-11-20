"""Build the TensorRT engine for the person/pose detector (NVIDIA GPUs; Jetson).

    python scripts/build_trt_engines.py            # the configured detector
    python scripts/build_trt_engines.py --force    # rebuild

Writes models/<detector>.engine: FP16, one dynamic engine covering every
input size the tracker uses (640 px, and 960 px for HD video), so tracking
continues across a size switch. An engine is tied to the GPU and TensorRT
version that built it, which is why it is built on the device, not shipped.
Takes ~10-15 min on a Jetson Orin Nano. The tracker picks it up when
`detector_engine` is on (the `jetson:` profile in config/model_config.yaml).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from src.tracking.person_tracker import resolve_weights  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not str(settings.device).startswith("cuda"):
        sys.exit("TensorRT needs an NVIDIA GPU; this machine runs on the CPU.")

    weights = Path(resolve_weights(settings.model.detector_weights))
    engine = weights.with_suffix(".engine")
    size = max(settings.model.detector_imgsz, settings.model.detector_imgsz_hd or 0)
    if engine.exists() and not a.force:
        print(f"{engine} exists (use --force to rebuild)")
        return

    from ultralytics import YOLO
    t0 = time.time()
    print(f"building {engine.name}: FP16, dynamic input up to {size} px ...", flush=True)
    # workspace=1: ultralytics sizes the dynamic profile's maximum as
    # workspace x imgsz, so 1 caps it at `size` (and the build memory at 1 GB,
    # which matters on a Jetson whose GPU memory is the system RAM).
    out = YOLO(str(weights)).export(format="engine", imgsz=size, half=True, dynamic=True,
                                    batch=1, workspace=1, simplify=False, device=0)
    weights.with_suffix(".onnx").unlink(missing_ok=True)      # intermediate
    print(f"built {out} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
