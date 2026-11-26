"""Models compiled for this machine's CPU by scripts/optimise_for_device.py.

On a machine without an NVIDIA GPU (a Raspberry Pi 5, a Rockchip board, an
Intel or AMD mini PC, a laptop), PyTorch is the slowest way to run the two
models that work all the time. setup.sh compiles them for the CPU it runs on,
measures every runtime, checks that each gives the same answers as PyTorch,
and keeps the fastest. Which one wins depends on the CPU; on a 6-core ARM
CPU (Cortex-A78AE: the Jetson's CPU, standing in for an ARM board), 4 threads:

                               PyTorch    OpenVINO    ONNX Runtime
    person detector, 416 px    217 ms     54 ms        -
    activity model (S3D)       1844 ms    30604 ms     530 ms

(OpenVINO compiles 3D convolutions poorly on ARM; on Intel it is usually
the fastest.) The compiled files live in models/compiled/ with
manifest.json, which records what was measured. They are made per machine
and not kept in git. Nothing here is used on an NVIDIA GPU (TensorRT and
CUDA do that job) or with CPU_RUNTIME=pytorch; a compiled model that fails
to load is skipped with a message, and the PyTorch model runs instead.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from config import settings

DIR = settings.models_dir / "compiled"
MANIFEST = DIR / "manifest.json"


def enabled() -> bool:
    return (not str(settings.device).startswith("cuda")
            and os.getenv("CPU_RUNTIME", "auto").split("#")[0].strip().lower() != "pytorch")


def manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text())
    except (OSError, ValueError):
        return {}


def threads() -> int:
    """Threads per model: all cores but leave room for the video (the
    detector, decoding and the activity model run side by side)."""
    return max(1, min(4, (os.cpu_count() or 4) - 1))


def detector_for(weights: str, imgsz: int) -> str | None:
    """The compiled detector for these weights at this input size, if any."""
    if not enabled():
        return None
    d = manifest().get("detector") or {}
    path = settings.root / d.get("path", "-")
    if d.get("weights") != Path(weights).name or not path.exists():
        return None
    if int(d.get("imgsz", 0)) != int(imgsz):
        print(f"[detector] compiled for {d.get('imgsz')} px but {imgsz} px is configured: "
              f"using PyTorch (re-run scripts/optimise_for_device.py)", flush=True)
        return None
    return str(path)


def load_activity(path: Path, runtime: str):
    """A function clip (1,3,16,224,224 float32) -> (logits, features)."""
    import numpy as np
    if runtime == "onnxruntime":
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads()
        so.log_severity_level = 3
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        return lambda x: tuple(np.asarray(o, np.float32) for o in sess.run(None, {"clip": x}))
    if runtime == "openvino":
        import openvino as ov
        core = ov.Core()
        cm = core.compile_model(core.read_model(str(path)), "CPU",
                                {"INFERENCE_NUM_THREADS": threads(), "INFERENCE_PRECISION_HINT": "f32"})
        lo, fe = cm.output("logits"), cm.output("feat")

        def run(x):
            r = cm([x])
            return np.asarray(r[lo], np.float32), np.asarray(r[fe], np.float32)
        return run
    raise ValueError(f"unknown runtime {runtime!r}")


def activity_model(backend: str):
    """(run function, runtime name) for the compiled activity model, or None."""
    if not enabled():
        return None
    a = (manifest().get("activity") or {}).get(backend) or {}
    path = settings.root / a.get("path", "-")
    if a.get("runtime") in (None, "pytorch") or not path.is_file():
        return None
    try:
        return load_activity(path, a["runtime"]), a["runtime"]
    except Exception as e:
        print(f"[deep] compiled {backend} model unusable ({str(e)[:100]}); using PyTorch", flush=True)
        return None
