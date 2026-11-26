#!/usr/bin/env bash
# One-time setup on Linux: any edge board or computer. No sudo needed.
#
#     ./setup.sh
#
# Then ./run.sh (dashboard on http://<this-machine's-address>:8000).
#
# It looks at the machine and installs what fits it:
#   NVIDIA Jetson               -> setup_jetson.sh (CUDA + TensorRT from JetPack)
#   PC / server, NVIDIA GPU     -> PyTorch with CUDA, the GPU profile
#   anything else (Raspberry Pi 5, Rockchip RK3588, Intel / AMD mini PC, ...)
#                               -> CPU-only PyTorch (no 3 GB of unusable CUDA
#                                  libraries), the CPU profile, and the person
#                                  detector compiled for this CPU
# Needs: 64-bit Linux (x86_64 or arm64), 4 GB RAM or more, ~4 GB of disk
# (10 GB with an NVIDIA GPU), internet during setup only.
# Re-running it is safe: finished steps are skipped or refreshed.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"
say() { printf '\n== %s\n' "$*"; }
die() { printf '\nSETUP STOPPED: %s\n' "$*" >&2; exit 1; }

# JETSON=0 installs the CPU profile even on a Jetson (to try how a board
# without an NVIDIA GPU behaves).
if [ -f /etc/nv_tegra_release ] && [ "${JETSON:-1}" != 0 ]; then
  exec ./setup_jetson.sh "$@"
fi

# ---------------------------------------------------------------------------
say "1/7  This machine"
[ "$(uname -s)" = Linux ] || die "this script is for Linux (Windows: setup.bat; macOS: README, 'Linux or macOS')."
ARCH=$(uname -m)
case "$ARCH" in
  x86_64)        LFS_ARCH=amd64 ;;
  aarch64|arm64) LFS_ARCH=arm64 ;;
  *) die "$ARCH is not supported: PyTorch needs a 64-bit OS (x86_64 or arm64). On a Raspberry Pi, install the 64-bit Raspberry Pi OS." ;;
esac
BOARD=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || true)
CPU=$(awk -F': ' '/model name/{print $2; exit}' /proc/cpuinfo 2>/dev/null || true)
RAM_GB=$(awk '/MemTotal/{printf "%.1f", $2/1048576}' /proc/meminfo)
FREE_GB=$(df -Pk . | awk 'NR==2{printf "%d", $4/1048576}')
GPU=""
if [ ! -f /etc/nv_tegra_release ] && command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
fi
echo "  ${BOARD:-${CPU:-$ARCH}}, $ARCH, $(nproc) cores, ${RAM_GB} GB RAM, ${FREE_GB} GB free disk"
if [ -n "$GPU" ]; then echo "  NVIDIA GPU: $GPU"; else echo "  no NVIDIA GPU: the CPU profile"; fi
awk "BEGIN{exit !($RAM_GB < 3.5)}" && echo "  WARNING: under 4 GB of RAM; the monitor may run out of memory."
NEED=4; [ -n "$GPU" ] && NEED=10
[ "$FREE_GB" -ge "$NEED" ] || die "only ${FREE_GB} GB free on this disk; ${NEED} GB needed."

# ---------------------------------------------------------------------------
say "2/7  Large files (models, demo clips) from Git LFS"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v git-lfs >/dev/null; then
  # The official release binary, checksum-verified (a package manager would need sudo).
  V=3.8.1; T=$(mktemp -d)
  F="git-lfs-linux-$LFS_ARCH-v$V.tar.gz"
  curl -fsSL -o "$T/$F" "https://github.com/git-lfs/git-lfs/releases/download/v$V/$F"
  SUM=$(curl -fsSL "https://github.com/git-lfs/git-lfs/releases/download/v$V/sha256sums.asc" | grep " \*\?$F\$" | awk '{print $1}')
  (cd "$T" && echo "$SUM  $F" | sha256sum -c -)
  tar -xzf "$T/$F" -C "$T"; mkdir -p "$HOME/.local/bin"
  cp "$T"/git-lfs-*/git-lfs "$HOME/.local/bin/"; rm -rf "$T"
fi
if [ -d .git ]; then
  git lfs install --local >/dev/null
  git lfs pull
fi
if [ "$(stat -c %s models/yolov8n-pose.pt 2>/dev/null || echo 0)" -lt 1000 ]; then
  die "models/*.pt are missing or Git LFS pointers. Clone the repository with git (not 'Download ZIP'), then run ./setup.sh again."
fi

# ---------------------------------------------------------------------------
say "3/7  Python 3.10-3.12 environment (.venv)"
ok_py() { "$1" -c 'import sys; sys.exit(not (3,10) <= sys.version_info[:2] <= (3,12))' 2>/dev/null; }
if [ -x .venv/bin/python ] && ! ok_py .venv/bin/python; then
  echo "  .venv has an unsupported Python; recreating it"; rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
  PY=""
  if [ "${USE_UV:-0}" != 1 ]; then
    for c in ${PYTHON:-} python3.11 python3.12 python3.10 python3; do
      if command -v "$c" >/dev/null && ok_py "$c" && "$c" -m venv .venv 2>/dev/null; then PY=$c; break; fi
      rm -rf .venv
    done
  fi
  if [ -z "$PY" ]; then
    # No usable system Python (Debian 13 and Raspberry Pi OS "Trixie" ship
    # 3.13; some images lack the venv module): uv brings its own Python 3.11,
    # into ~/.local, no sudo.
    echo "  no Python 3.10-3.12 with venv on this system; fetching Python 3.11 with uv"
    if ! command -v uv >/dev/null; then
      V=0.8.0; T=$(mktemp -d)
      F="uv-$( [ "$LFS_ARCH" = arm64 ] && echo aarch64 || echo x86_64 )-unknown-linux-gnu.tar.gz"
      curl -fsSL -o "$T/$F" "https://github.com/astral-sh/uv/releases/download/$V/$F"
      SUM=$(curl -fsSL "https://github.com/astral-sh/uv/releases/download/$V/$F.sha256" | awk '{print $1}')
      (cd "$T" && echo "$SUM  $F" | sha256sum -c -)
      tar -xzf "$T/$F" -C "$T"; mkdir -p "$HOME/.local/bin"
      cp "$T"/uv-*/uv "$HOME/.local/bin/"; rm -rf "$T"
    fi
    uv venv --seed --python 3.11 .venv
  fi
fi
.venv/bin/python --version
PIP=(.venv/bin/python -m pip install --disable-pip-version-check)
"${PIP[@]}" -q --upgrade pip wheel

# ---------------------------------------------------------------------------
say "4/7  Packages"
if [ -n "$GPU" ] && [ "$ARCH" = x86_64 ]; then
  "${PIP[@]}" -r requirements-gpu.txt
elif [ -n "$GPU" ]; then
  # An arm64 server with an NVIDIA GPU (Grace Hopper...): PyPI's PyTorch has CUDA there.
  "${PIP[@]}" -r requirements.txt
else
  # The CPU-only build: PyPI's default PyTorch now carries ~3 GB of CUDA
  # libraries, even on arm64 boards that can never use them.
  "${PIP[@]}" torch torchvision --index-url https://download.pytorch.org/whl/cpu
  "${PIP[@]}" -r requirements.txt -r requirements-edge.txt
fi
"${PIP[@]}" -q imageio-ffmpeg==0.8.1
ln -sf "$(.venv/bin/python -c 'import imageio_ffmpeg as f; print(f.get_ffmpeg_exe())')" .venv/bin/ffmpeg
if ! ERR=$(.venv/bin/python -c "import cv2" 2>&1); then
  echo "$ERR" | tail -1
  die "OpenCV needs two system libraries here. Install them once, then run ./setup.sh again:
    sudo apt install -y libgl1 libglib2.0-0"
fi

# ---------------------------------------------------------------------------
say "5/7  Model weights for this machine"
.venv/bin/python scripts/download_models.py

# ---------------------------------------------------------------------------
say "6/7  Compiling the person detector for this machine"
.venv/bin/python scripts/optimise_for_device.py || echo "  (skipped: the monitor uses the standard detector)"

# ---------------------------------------------------------------------------
say "7/7  Checks"
[ -f .env ] || { cp .env.example .env; echo "  created .env from .env.example"; }
.venv/bin/python scripts/verify_install.py
echo
echo "Setup done. Start the monitor:   ./run.sh"
echo "Start it at every boot:          scripts/install_service.sh"
