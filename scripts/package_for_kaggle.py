"""Build the zip to upload to Kaggle as a (private) Dataset.

    .venv\\Scripts\\python.exe scripts\\package_for_kaggle.py
    .venv\\Scripts\\python.exe scripts\\package_for_kaggle.py --with-phone-clips

Produces dist/human-action-monitor.zip containing exactly what the app needs
to run: code, config, bundled model weights, the pre-built dashboard and the
demo clips. Left out: the local virtual environment, logs and database,
backups, experiment runs, node_modules, raw datasets and office documents.

Personal phone recordings (WhatsApp_*) are left out unless asked for, since
an uploaded dataset leaves this machine. Kaggle datasets are private by
default; keep it that way.
"""
from __future__ import annotations

import argparse
import time
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INCLUDE = ["config", "src", "scripts", "models", "kaggle", "docs", "tests",
           "ui/dist", "data/samples", "data/showcase", "data/candidates/720p",
           "requirements.txt", "requirements-gpu.txt", "requirements-optional.txt",
           "README.md", ".env.example", "Dockerfile", "docker-compose.yml", ".dockerignore"]
SKIP_DIRS = {"__pycache__", ".pytest_cache", "node_modules", ".venv", "checkpoints_tmp"}
SKIP_SUFFIX = {".pyc", ".docx", ".part", ".log", ".db"}


def wanted(p: Path, phone: bool) -> bool:
    if any(part in SKIP_DIRS for part in p.parts) or p.suffix.lower() in SKIP_SUFFIX:
        return False
    if p.name.startswith(".") and p.name not in {".env.example", ".dockerignore", ".gitkeep"}:
        return False
    if not phone and p.name.startswith("WhatsApp_"):
        return False
    # HD benchmark footage: skip the odd very large file to keep the upload lean.
    if "candidates" in p.parts and (ROOT / p).stat().st_size > 20 * 2**20:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-phone-clips", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "human-action-monitor.zip")
    args = ap.parse_args()

    if not (ROOT / "ui" / "dist" / "index.html").exists():
        sys.exit("ui/dist is missing - build the dashboard first: cd ui && npm run build")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for item in INCLUDE:
        p = ROOT / item
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files += [f for f in p.rglob("*") if f.is_file()]
    files = [f for f in files if wanted(f.relative_to(ROOT), args.with_phone_clips)]

    total = 0
    # Videos and weights are already compressed; storing them is as small
    # and far faster than deflating them again.
    # A marker unique to this build. Its NAME is visible in Kaggle's file
    # list, so the uploader can wait until this exact version is live:
    # right after a new version is submitted, Kaggle still reports the
    # previous one as "ready", and a job started then gets the old files.
    build_id = time.strftime("%Y%m%d-%H%M%S")
    with zipfile.ZipFile(args.out, "w") as z:
        z.writestr(f"human-action-monitor/BUILD-{build_id}.txt", build_id)
        for f in files:
            rel = f.relative_to(ROOT).as_posix()
            stored = f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".pt", ".pth"}
            z.write(f, f"human-action-monitor/{rel}",
                    compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
            total += f.stat().st_size
    (args.out.parent / "build_id.txt").write_text(build_id)

    size = args.out.stat().st_size / 2**20
    print(f"\n  {len(files)} files, {total / 2**20:.0f} MB -> {args.out} ({size:.0f} MB)")
    print("\n  Next: kaggle.com -> Create -> New Dataset -> upload this zip (keep it Private).")
    print("  Then open kaggle/kaggle_demo.ipynb on Kaggle and add the dataset as input.\n")


if __name__ == "__main__":
    main()
