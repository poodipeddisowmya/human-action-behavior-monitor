"""Fetch pretrained weights.

YOLOv8n is auto-downloaded by ultralytics on first use, so this script simply
'warms' it. Extend when you enable ST-GCN or VideoMAE weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from config import settings


def main() -> None:
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. YOLO (detection + pose in one) ---
    print(f"[1/2] Warming detector weights ({settings.model.detector_weights})...")
    YOLO(settings.model.detector_weights)
    print("      done.")

    # --- 2. VideoMAE (Kinetics-400 action classifier) ---
    if (settings.model.action_classifier or "").lower() == "videomae":
        print("[2/2] Downloading VideoMAE weights (~350 MB) — first run only...")
        try:
            from transformers import (
                VideoMAEForVideoClassification,
                VideoMAEImageProcessor,
            )
            ckpt = "MCG-NJU/videomae-base-finetuned-kinetics"
            VideoMAEImageProcessor.from_pretrained(ckpt)
            VideoMAEForVideoClassification.from_pretrained(ckpt)
            print("      done.")
        except ImportError:
            print("      transformers not installed. Run: pip install -r requirements.txt")
        except Exception as e:
            print(f"      failed: {e}")
            print("      pipeline will fall back to the heuristic classifier.")
    else:
        print(f"[2/2] action_classifier={settings.model.action_classifier} — "
              "no deep-model weights to fetch.")

    stgcn_path: Path = settings.models_dir / "st_gcn_kinetics.pt"
    if not stgcn_path.exists():
        print(
            f"\n[optional] Pose-based ST-GCN weights not present at {stgcn_path}.\n"
            "           Only needed if you set action_classifier: st_gcn_kinetics."
        )


if __name__ == "__main__":
    main()
