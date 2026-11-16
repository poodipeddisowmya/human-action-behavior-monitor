#!/usr/bin/env bash
# Start the monitor on a Jetson:  ./run_jetson.sh [--source <file|camera>]
# Dashboard: http://localhost:8000 (or http://<jetson-ip>:8000 on the LAN).
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$PWD/.venv/bin:$PATH" PYTHONIOENCODING=utf-8

# CPU and GPU share one pool of RAM on a Jetson, and CUDA allocations do not
# evict the page cache: with the cache full, loading the models fails with
# "NvMapMemAlloc error 12" / "NVML_SUCCESS == r INTERNAL ASSERT FAILED" even
# though `free` shows gigabytes "available". Touching a large buffer once
# makes the kernel evict the cache without needing sudo for drop_caches.
python -c "import numpy as np; np.ones(int(${JETSON_RECLAIM_GB:-2}*2**30)//8).sum()" || true

exec python scripts/run_pipeline.py "$@"
