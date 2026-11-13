"""Benchmark candidate action-recognition backends on this machine.

Measures real per-inference latency for the models we could run, so the
choice of backend is made from numbers rather than intuition.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WARMUP, RUNS = 2, 8
CLIP = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)


def timeit(fn, warmup=WARMUP, runs=RUNS) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - t0) / runs * 1000  # ms


def bench_videomae(quantize: bool):
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
    ck = "MCG-NJU/videomae-base-finetuned-kinetics"
    proc = VideoMAEImageProcessor.from_pretrained(ck)
    model = VideoMAEForVideoClassification.from_pretrained(ck).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if quantize:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    inputs = proc(list(CLIP), return_tensors="pt")

    def run():
        with torch.no_grad():
            model(**inputs)

    return timeit(run), n_params


def bench_s3d(quantize: bool):
    from torchvision.models.video import S3D_Weights, s3d
    w = S3D_Weights.KINETICS400_V1
    model = s3d(weights=w).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if quantize:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    # s3d expects (B, C, T, H, W) float, normalized
    x = torch.from_numpy(CLIP).float().div(255).permute(3, 0, 1, 2).unsqueeze(0)

    def run():
        with torch.no_grad():
            model(x)

    return timeit(run), n_params


def bench_mc3(quantize: bool):
    from torchvision.models.video import MC3_18_Weights, mc3_18
    model = mc3_18(weights=MC3_18_Weights.KINETICS400_V1).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if quantize:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    x = torch.from_numpy(CLIP).float().div(255).permute(3, 0, 1, 2).unsqueeze(0)

    def run():
        with torch.no_grad():
            model(x)

    return timeit(run), n_params


def bench_yolo():
    from ultralytics import YOLO
    m = YOLO("yolov8n-pose.pt")
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def run():
        m.predict(frame, verbose=False, device="cpu")

    return timeit(run), 3.3


def main() -> None:
    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}\n")
    rows = []

    for name, fn in [
        ("YOLOv8n-pose (per frame)", lambda: bench_yolo()),
        ("VideoMAE-base fp32",       lambda: bench_videomae(False)),
        ("VideoMAE-base int8",       lambda: bench_videomae(True)),
        ("s3d fp32",                 lambda: bench_s3d(False)),
        ("s3d int8",                 lambda: bench_s3d(True)),
        ("mc3_18 fp32",              lambda: bench_mc3(False)),
    ]:
        try:
            ms, params = fn()
            rows.append((name, params, ms))
            print(f"  {name:<28} {params:>6.1f}M params   {ms:>8.1f} ms")
        except Exception as e:
            print(f"  {name:<28} FAILED: {str(e)[:70]}")

    if rows:
        base = next((r for r in rows if r[0].startswith("VideoMAE-base fp32")), None)
        if base:
            print(f"\n  speedup vs VideoMAE-base fp32 ({base[2]:.0f} ms):")
            for name, _, ms in rows:
                if "YOLO" in name or name == base[0]:
                    continue
                print(f"    {name:<28} {base[2]/ms:>5.1f}x faster")


if __name__ == "__main__":
    main()
