"""Micro-benchmark: measures FPS, RAM, GPU on a fixed video."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import PerformanceMonitor
from src.input import FileSource
from src.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--seconds", type=int, default=30)
    args = parser.parse_args()

    perf = PerformanceMonitor()
    pipeline = Pipeline(
        FileSource(args.source),
        on_frame=lambda *_: perf.tick(),
        on_event=lambda e: None,
    )
    pipeline.start()

    time.sleep(args.seconds)
    pipeline.stop()

    snap = perf.snapshot()
    print("\n===== benchmark =====")
    for k, v in snap.items():
        print(f"{k:>10}: {v}")


if __name__ == "__main__":
    main()
