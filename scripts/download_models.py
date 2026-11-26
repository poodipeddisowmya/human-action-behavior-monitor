"""Download the model weights this machine will use, so the monitor then
runs fully offline.

    python scripts/download_models.py          # what this machine's profile needs
    python scripts/download_models.py --all    # every model, e.g. for an offline image

setup.sh, setup_jetson.sh and the Dockerfile run it. Bundled in the
repository already (Git LFS): the person/pose detectors, the S3D activity
model and the face models. Downloaded here, into the Hugging Face cache
(~/.cache/huggingface, or HF_HOME), depending on the profile
(config/model_config.yaml):

    CLIP        places in the subtitle ("in an office")     every machine, ~600 MB
    VideoMAE    activity model on an NVIDIA GPU              ~350 MB
    X-CLIP      activities described in words (open_vocab)  NVIDIA GPU, ~800 MB
    SmolVLM2    detailed subtitles (vlm_captions)            NVIDIA GPU, ~1 GB

A model that fails to download is reported and skipped: the monitor runs
without that one feature (and says so in its log).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["HF_HUB_OFFLINE"] = "0"           # this script is the one that goes online

from config import settings  # noqa: E402


def wanted(everything: bool) -> list[tuple[str, str, str]]:
    """(what, model class, processor class, checkpoint) this machine uses."""
    gpu = str(settings.device).startswith("cuda")
    m = settings.model
    out = []
    if everything or m.scene_description:
        out.append(("CLIP (places)", "CLIPModel", "CLIPProcessor", "openai/clip-vit-base-patch32"))
    if everything or m.deep_backend == "videomae":
        from src.action_recognition.clip_encoder import VMAE_CHECKPOINT
        out.append(("VideoMAE (activities)", "VideoMAEForVideoClassification",
                    "VideoMAEImageProcessor", VMAE_CHECKPOINT))
    if everything or (gpu and settings.open_vocab.get("enabled", True)):
        out.append(("X-CLIP (activities in words)", "XCLIPModel", "XCLIPProcessor",
                    str(settings.open_vocab.get("model", "microsoft/xclip-base-patch16-zero-shot"))))
    if everything or (gpu and m.vlm_captions):
        from src.description.narrator import CHECKPOINT
        out.append(("SmolVLM2 (detailed subtitles)", "AutoModelForImageTextToText",
                    "AutoProcessor", CHECKPOINT))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all", action="store_true", help="every model, whatever this machine uses")
    a = ap.parse_args()

    import transformers
    from ultralytics import YOLO

    # Bundled, but a missing one is fetched here rather than on first start.
    for w in {settings.model.detector_weights, settings.model.detector_fallback_weights}:
        p = settings.models_dir / w
        if not p.is_file() or p.stat().st_size < 1000:
            print(f"  detector {w}: downloading")
            YOLO(str(p))
    from torchvision.models.video import S3D_Weights, s3d
    s3d(weights=S3D_Weights.KINETICS400_V1)          # bundled in models/torch (TORCH_HOME)

    failed = []
    for what, model_cls, proc_cls, ckpt in wanted(a.all):
        print(f"  {what}: {ckpt}", flush=True)
        try:
            getattr(transformers, proc_cls).from_pretrained(ckpt)
            getattr(transformers, model_cls).from_pretrained(ckpt)
        except Exception as e:
            print(f"    failed ({str(e)[:120]}); the monitor runs without it")
            failed.append(what)
    print("models ready" + (f", except: {', '.join(failed)}" if failed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
