#!/usr/bin/env bash
# Start the monitor on Linux: a Jetson, a PC or server, or any edge board.
#
#     ./run.sh                         # what .env's SOURCE says, else the first demo clip
#     ./run.sh --source 0              # USB camera
#     ./run.sh --source Reception      # a CCTV camera named in .env
#
# Dashboard: http://localhost:8000, or http://<this-machine's-address>:8000.
# Install first with ./setup.sh. To start at boot: scripts/install_service.sh
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Not installed yet: run ./setup.sh first."
  exit 1
fi
export PATH="$PWD/.venv/bin:$PATH" PYTHONIOENCODING=utf-8

if [ -f /etc/nv_tegra_release ]; then
  # Jetson: CPU and GPU share one pool of RAM, and CUDA allocations do not
  # evict the page cache: with the cache full, loading the models fails with
  # "NvMapMemAlloc error 12" / "NVML_SUCCESS == r INTERNAL ASSERT FAILED"
  # even though `free` shows gigabytes "available". Touching a large buffer
  # once makes the kernel evict the cache, without sudo for drop_caches.
  python -c "import numpy as np; np.ones(int(${JETSON_RECLAIM_GB:-2}*2**30)//8).sum()" || true
fi

exec python scripts/run_pipeline.py "$@"
