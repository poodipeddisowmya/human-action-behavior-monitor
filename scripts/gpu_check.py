"""Is the GPU being used, and how fast is each model on it?

    .venv\\Scripts\\python.exe scripts\\gpu_check.py

Prints the device the app will use, the GPU and its memory, and the time per
call of each model the live pipeline runs, so a new machine can be checked
in under a minute before a demo.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from config import settings  # noqa: E402


def timed(fn, n=10):
    fn()                                   # warm-up (first call loads kernels)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main() -> None:
    print(f"\n  torch {torch.__version__}, CUDA build: {torch.version.cuda or 'none (CPU-only build)'}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name}, {p.total_memory / 2**30:.1f} GB")
    else:
        print("  GPU: not available — the app will run on the CPU.")
        print("  (If this machine has an NVIDIA GPU, run setup_gpu.bat to install the GPU build.)")
    print(f"  App will use: {settings.device}  |  activity look every "
          f"{settings.model.deep_min_interval_s}s  |  detector {settings.model.detector_imgsz}px\n")

    frame = (np.random.rand(540, 960, 3) * 255).astype(np.uint8)

    from ultralytics import YOLO
    from src.tracking.person_tracker import resolve_weights
    pose = YOLO(resolve_weights(settings.model.detector_weights))
    ms = timed(lambda: pose.predict(frame, imgsz=settings.model.detector_imgsz,
                                    device=settings.device, verbose=False))
    print(f"  person + pose detector : {ms:6.1f} ms per frame")

    from torchvision.models.video import S3D_Weights, s3d
    m = s3d(weights=S3D_Weights.KINETICS400_V1).eval().to(settings.device)
    clip = torch.rand(1, 3, 16, 224, 224, device=settings.device)
    with torch.no_grad():
        ms = timed(lambda: m(clip))
    print(f"  activity model (S3D)   : {ms:6.1f} ms per look")

    try:
        from src.description.clip_shared import encode_images
        img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        ms = timed(lambda: encode_images(img))
        print(f"  place / posture (CLIP) : {ms:6.1f} ms per look")
    except Exception as e:
        print(f"  place / posture (CLIP) : unavailable ({e})")

    if torch.cuda.is_available():
        print(f"\n  GPU memory in use: {torch.cuda.memory_allocated() / 2**20:.0f} MB")
    print()


if __name__ == "__main__":
    main()
