"""What machine is this? One line for logs, setup and the install check."""
from __future__ import annotations

import os
import platform
from pathlib import Path


def _read(path: str) -> str:
    try:
        return Path(path).read_text(errors="ignore").replace("\0", "").strip()
    except OSError:
        return ""


def describe() -> dict:
    """Board / CPU / RAM / accelerator, as far as the OS tells."""
    cpu = ""
    for line in _read("/proc/cpuinfo").splitlines():
        if line.lower().startswith(("model name", "hardware")):
            cpu = line.split(":", 1)[1].strip()
            break
    if not cpu:
        cpu = platform.processor() or platform.machine()
    ram_gb = 0.0
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("MemTotal"):
            ram_gb = int(line.split()[1]) / 1048576
    if not ram_gb and hasattr(os, "sysconf"):
        try:
            ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        except (ValueError, OSError):
            pass
    jetson = Path("/etc/nv_tegra_release").exists()
    return {
        "board": _read("/proc/device-tree/model"),
        "cpu": cpu,
        "arch": platform.machine(),
        "cores": os.cpu_count() or 1,
        "ram_gb": round(ram_gb, 1),
        "os": f"{platform.system()} {platform.release()}",
        "kind": "jetson" if jetson else "pc",
    }


def summary(device: str | None = None) -> str:
    d = describe()
    name = d["board"] or d["cpu"] or d["arch"]
    s = f"{name}, {d['arch']}, {d['cores']} cores, {d['ram_gb']:.0f} GB RAM"
    if device:
        s += f", running on {'the GPU' if str(device).startswith('cuda') else 'the CPU'}"
    return s
