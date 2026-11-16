#!/usr/bin/env bash
# One-time setup on an NVIDIA Jetson (JetPack 6.x). No sudo needed.
#
#     ./setup_jetson.sh
#
# Then:  ./run_jetson.sh            (dashboard on http://<jetson-ip>:8000)
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\n== %s\n' "$*"; }

say "1/6  Checking the board"
cat /proc/device-tree/model 2>/dev/null | tr -d '\0'; echo
head -1 /etc/nv_tegra_release 2>/dev/null || echo "(not a Jetson? continuing)"
PY=python3.10
command -v $PY >/dev/null || { echo "Python 3.10 not found (JetPack 6 ships it)"; exit 1; }

say "2/6  Large files (weights, demo clips) from Git LFS"
if ! command -v git-lfs >/dev/null && [ ! -x "$HOME/.local/bin/git-lfs" ]; then
  # Official release binary, checksum-verified; apt would need sudo.
  V=3.8.1; T=$(mktemp -d)
  curl -sSL -o "$T/lfs.tgz" "https://github.com/git-lfs/git-lfs/releases/download/v$V/git-lfs-linux-arm64-v$V.tar.gz"
  SUM=$(curl -sSL "https://github.com/git-lfs/git-lfs/releases/download/v$V/sha256sums.asc" \
        | grep "linux-arm64-v$V.tar.gz" | awk '{print $1}')
  echo "$SUM  $T/lfs.tgz" | sha256sum -c -
  tar -xzf "$T/lfs.tgz" -C "$T"; mkdir -p "$HOME/.local/bin"
  cp "$T"/git-lfs-*/git-lfs "$HOME/.local/bin/"; rm -rf "$T"
fi
export PATH="$HOME/.local/bin:$PATH"
git lfs install --local >/dev/null
git lfs pull
if [ "$(stat -c %s models/yolov8n-pose.pt)" -lt 1000 ]; then
  echo "models/*.pt are still LFS pointers - git lfs pull failed"; exit 1
fi

say "3/6  Python environment (.venv)"
if [ ! -x .venv/bin/python ]; then
  # python3-venv (ensurepip) is not installed on JetPack by default.
  $PY -m pip install --user -q virtualenv
  $PY -m virtualenv -q .venv
fi
.venv/bin/python -m pip install -q --upgrade pip wheel
.venv/bin/pip install -r requirements-jetson.txt
FF=$(.venv/bin/python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
ln -sf "$FF" .venv/bin/ffmpeg
# JetPack's TensorRT Python bindings (python3-libnvinfer) live in the system
# site-packages; link just those into the venv.
SP=$(.venv/bin/python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
for p in /usr/lib/python3.10/dist-packages/tensorrt*; do ln -sfn "$p" "$SP/$(basename "$p")"; done
.venv/bin/python -c "import tensorrt as t; print('  TensorRT', t.__version__)"

say "4/6  Model weights (CLIP, VideoMAE, SmolVLM2)"
export HF_HUB_OFFLINE=0
.venv/bin/python - <<'EOF'
from transformers import (CLIPModel, CLIPProcessor, VideoMAEForVideoClassification,
                          VideoMAEImageProcessor, AutoModelForImageTextToText, AutoProcessor)
for M, P, c in [(CLIPModel, CLIPProcessor, "openai/clip-vit-base-patch32"),
                (VideoMAEForVideoClassification, VideoMAEImageProcessor,
                 "MCG-NJU/videomae-base-finetuned-kinetics"),
                (AutoModelForImageTextToText, AutoProcessor,
                 "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")]:
    M.from_pretrained(c); P.from_pretrained(c); print("  ready:", c)
EOF

export PATH="$PWD/.venv/bin:$PATH"
python -c "import numpy as np; np.ones(int(2*2**30)//8).sum()"   # room for the build

say "5/6  TensorRT engine for the pose detector (~10-15 min, once)"
python scripts/build_trt_engines.py

say "6/6  Checks"
python scripts/verify_install.py
python scripts/gpu_check.py || true
echo
echo "Setup done. Start the dashboard with:  ./run_jetson.sh"
