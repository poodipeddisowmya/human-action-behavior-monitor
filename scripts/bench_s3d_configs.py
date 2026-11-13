"""Find the s3d spatial resolution that hits real-time on this CPU.

s3d is fully convolutional spatially, so it accepts smaller frames than the
224x224 default. Clip length is NOT flexible: the network downsamples time by
8x and its final avg-pool needs kT=2, so T must stay >= 16.

Also sweeps YOLOv8-pose input size, which is the other half of the latency
budget (it runs on every frame, not just on classification strides).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def timeit(fn, warmup=2, runs=6) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - t0) / runs * 1000


def main() -> None:
    from torchvision.models.video import S3D_Weights, s3d
    model = s3d(weights=S3D_Weights.KINETICS400_V1).eval()

    print("s3d — spatial size (clip length fixed at 16)\n")
    print(f"  {'config':<20}{'latency':>12}{'speedup':>11}")
    print("  " + "-" * 44)

    base = None
    for hw in (224, 192, 160, 128, 112, 96):
        x = torch.randn(1, 3, 16, hw, hw)
        try:
            with torch.no_grad():
                ms = timeit(lambda: model(x))
        except Exception as e:
            print(f"  {f'{hw}x{hw}':<20}  unsupported ({str(e)[:28]})")
            continue
        if base is None:
            base = ms
        print(f"  {f'{hw}x{hw} x 16f':<20}{ms:>9.0f} ms{base/ms:>10.1f}x")

    print("\nYOLOv8n-pose — input size (runs on EVERY frame)\n")
    from ultralytics import YOLO
    m = YOLO("yolov8n-pose.pt")
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"  {'imgsz':<20}{'latency':>12}{'max fps':>12}")
    print("  " + "-" * 44)
    for sz in (640, 512, 416, 320):
        ms = timeit(lambda: m.predict(frame, imgsz=sz, verbose=False, device="cpu"))
        print(f"  {sz:<20}{ms:>9.0f} ms{1000/ms:>11.1f}")


if __name__ == "__main__":
    main()
