"""Fetch a small bundle of public, no-auth sample clips.

Each clip covers a different scenario the classifier handles differently, so a
demo can flip between them to show the system on varied inputs.

Anything real (a phone video, a security camera clip) will work better — this
is just a curated starter bundle so the demo runs without hunting for footage.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings

# name  ->  (url, local extension, one-line description)
# Two groups:
#   A) surveillance / walking-heavy clips — for detection + tracking demos
#   B) activity-rich clips — for VideoMAE action classification demos
BUNDLE: dict[str, tuple[str, str, str]] = {
    # --- Group A: person-detection / tracking demos ---
    "street-surveillance": (
        "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi",
        ".avi",
        "Top-down street view — pedestrians walking, occasional pauses.",
    ),
    "office-people": (
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4",
        ".mp4",
        "Office corridor — full-body walking, single person entering / leaving.",
    ),
    "store-aisle": (
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/store-aisle-detection.mp4",
        ".mp4",
        "Retail store aisle — multiple people, mixed walking / standing / browsing.",
    ),
    "worker-zone": (
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/worker-zone-detection.mp4",
        ".mp4",
        "Industrial floor — worker moving around equipment. Good for zone rules later.",
    ),
    "person-bicycle-car": (
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4",
        ".mp4",
        "Mixed scene — people plus vehicles (detector only tracks people).",
    ),

    # --- Group B: activity-rich clips (VideoMAE will fire clean labels) ---
    "activity-eating-spaghetti": (
        "https://huggingface.co/datasets/hf-internal-testing/spaghetti-video/resolve/main/eating_spaghetti.mp4",
        ".mp4",
        "HuggingFace VideoMAE test clip — clear eating action, close-up.",
    ),
    "activity-mmaction-demo": (
        "https://github.com/open-mmlab/mmaction2/raw/main/demo/demo.mp4",
        ".mp4",
        "MMAction2 official demo — rhythmic gymnastics / arm-wave activity.",
    ),
    "activity-yoga": (
        "https://assets.mixkit.co/videos/preview/mixkit-woman-doing-yoga-with-a-band-4830-large.mp4",
        ".mp4",
        "Mixkit — woman doing yoga poses. Should fire 'exercising' or 'yoga'.",
    ),
    "activity-cooking": (
        "https://assets.mixkit.co/videos/preview/mixkit-woman-mixing-a-bowl-of-ingredients-in-a-kitchen-42877-large.mp4",
        ".mp4",
        "Mixkit — woman cooking in a kitchen. Should fire 'cooking'.",
    ),
    "activity-dancing": (
        "https://assets.mixkit.co/videos/preview/mixkit-young-woman-dancing-happily-4702-large.mp4",
        ".mp4",
        "Mixkit — young woman dancing. Should fire 'dancing'.",
    ),
    "activity-reading": (
        "https://assets.mixkit.co/videos/preview/mixkit-woman-reading-a-book-in-a-cafe-4919-large.mp4",
        ".mp4",
        "Mixkit — woman reading a book. Should fire 'reading'.",
    ),
    "activity-exercising": (
        "https://assets.mixkit.co/videos/preview/mixkit-fitness-woman-doing-push-ups-27884-large.mp4",
        ".mp4",
        "Mixkit — fitness push-ups. Should fire 'exercising' or 'push-up'.",
    ),
}

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=45) as r, dest.open("wb") as f:
            while chunk := r.read(64 * 1024):
                f.write(chunk)
        return dest.stat().st_size > 50_000
    except Exception as e:
        print(f"    failed: {e}")
        return False


def _hf_download(repo_id: str, filename: str, dest: Path, repo_type: str = "dataset") -> bool:
    """Fetch a file from a HuggingFace dataset repo via the official client."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
        import shutil
        shutil.copyfile(path, dest)
        return dest.stat().st_size > 50_000
    except Exception as e:
        print(f"    hf_hub failed: {e}")
        return False


# HuggingFace-hosted activity clips known to exist. Same format as BUNDLE but
# fetched through hf_hub_download so redirects and gated logic work.
HF_BUNDLE: dict[str, tuple[str, str, str, str]] = {
    # (repo_id, filename, ext, description)
    "activity-eating": (
        "hf-internal-testing/fixtures_video",
        "eating_spaghetti.mp4",
        ".mp4",
        "HF test clip — clear eating action.",
    ),
    "activity-archery": (
        "sayakpaul/ucf101-subset",
        "v_Archery_g07_c04.avi",
        ".avi",
        "UCF101 archery clip — arm motion, focused activity.",
    ),
}


def main() -> None:
    samples = settings.data_dir / "samples"
    samples.mkdir(parents=True, exist_ok=True)

    print(f"Downloading demo bundle → {samples}\n")
    results: list[tuple[str, Path | None, str]] = []
    for name, (url, ext, desc) in BUNDLE.items():
        dest = samples / f"{name}{ext}"
        if dest.exists() and dest.stat().st_size > 50_000:
            print(f"  ✓ {name}{ext:<5}  (already present, {dest.stat().st_size // 1024} KB)")
            results.append((name, dest, desc))
            continue

        print(f"  → {name}{ext:<5}  fetching...")
        if _download(url, dest):
            print(f"    done ({dest.stat().st_size // 1024} KB)")
            results.append((name, dest, desc))
        else:
            results.append((name, None, desc))

    # HuggingFace-hosted clips
    print("\nHuggingFace-hosted clips:")
    for name, (repo, fname, ext, desc) in HF_BUNDLE.items():
        dest = samples / f"{name}{ext}"
        if dest.exists() and dest.stat().st_size > 50_000:
            print(f"  ✓ {name}{ext}  (already present)")
            results.append((name, dest, desc))
            continue
        print(f"  → {name}{ext}  fetching from HF ({repo})...")
        if _hf_download(repo, fname, dest):
            print(f"    done ({dest.stat().st_size // 1024} KB)")
            results.append((name, dest, desc))
        else:
            results.append((name, None, desc))

    # Keep demo.mp4 pointing at street-surveillance for backward-compat.
    default = samples / "street-surveillance.avi"
    if default.exists():
        legacy = samples / "demo.mp4"
        if legacy.exists():
            legacy.unlink()
        legacy.write_bytes(default.read_bytes())

    print("\nAvailable clips:")
    print(f"  {'name':<22} {'file':<28} description")
    print(f"  {'-'*22} {'-'*28} {'-'*40}")
    for name, path, desc in results:
        file_col = path.name if path else "(download failed)"
        print(f"  {name:<22} {file_col:<28} {desc}")

    ok = [name for name, path, _ in results if path]
    print(f"\n{len(ok)}/{len(BUNDLE)} clips ready.")
    print("Run one with:")
    print(f"  python scripts\\run_pipeline.py --source data\\samples\\{ok[0] if ok else '<name>'}.mp4")
    print("Or switch live from the UI header dropdown once the backend is running.")


if __name__ == "__main__":
    main()
