"""Make the monitor as fast as it can be on this machine. setup.sh runs it.

    python scripts/optimise_for_device.py            # measure, compile, keep the fastest
    python scripts/optimise_for_device.py --remove   # back to plain PyTorch

NVIDIA GPU: the GPU already runs the models with CUDA; on a Jetson this
builds the TensorRT engine for the person detector (build_trt_engines.py).

CPU (any edge board or PC without an NVIDIA GPU): compiles the two models
that run all the time, the person detector and the activity model (S3D),
with each runtime available (OpenVINO, ONNX Runtime), times each against
PyTorch on real frames from the demo clips, checks that it gives the same
answers, and keeps the fastest. Which runtime wins depends on the CPU
(src/utils/compiled_models.py has the measurements), which is why it is
measured here rather than decided in advance. Takes 2-10 minutes; the result
goes to models/compiled/ and is used from the next start. Re-run it after
changing detector_weights or detector_imgsz in config/model_config.yaml.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config import settings  # noqa: E402
from config.settings import is_jetson  # noqa: E402
from src.utils import compiled_models as cm  # noqa: E402
from src.utils.hardware import describe, summary  # noqa: E402

ROOT = settings.root
FASTER = 0.9            # keep a compiled model only if it takes <= 90% of PyTorch's time


def say(msg: str = "") -> None:
    print(msg, flush=True)


def demo_frames(n_clips: int = 4, per_clip: int = 6) -> list[np.ndarray]:
    clips = sorted((ROOT / "data" / "samples").glob("*.mp4"))[:n_clips]
    frames = []
    for c in clips:
        cap = cv2.VideoCapture(str(c))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for i in np.linspace(10, max(11, total - 10), per_clip).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(f)
        cap.release()
    if not frames:
        frames = [np.full((480, 640, 3), 128, np.uint8)]
    return frames


def timed(fn, inputs, repeat: int = 1, budget_s: float = 60.0) -> float:
    """Median ms per call, after one warm-up; stops early past the budget."""
    fn(inputs[0])
    times, start = [], time.perf_counter()
    for _ in range(repeat):
        for x in inputs:
            t = time.perf_counter()
            fn(x)
            times.append(time.perf_counter() - t)
            if time.perf_counter() - start > budget_s:
                return float(np.median(times)) * 1000
    return float(np.median(times)) * 1000


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def same_people(ref: list, got: list) -> float:
    """Share of people found by either model that both found (IoU >= 0.5)."""
    matched, used = 0, set()
    for r in ref:
        best = max(((i, _iou(r, g)) for i, g in enumerate(got) if i not in used),
                   key=lambda t: t[1], default=(None, 0))
        if best[1] >= 0.5:
            matched += 1
            used.add(best[0])
    total = len(ref) + len(got) - matched
    return matched / total if total else 1.0


# ---------------------------------------------------------------------------
def optimise_detector(frames, out: Path) -> dict | None:
    from ultralytics import YOLO
    from src.tracking.person_tracker import resolve_weights
    weights = Path(resolve_weights(settings.model.detector_weights))
    imgsz = int(settings.model.detector_imgsz)
    conf = settings.model.detector_conf

    def boxes(model):
        return lambda f: model.predict(f, imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]

    ref_model = YOLO(str(weights), task="pose")
    ref = [boxes(ref_model)(f).boxes.xyxy.tolist() for f in frames]
    pt_ms = timed(boxes(ref_model), frames)
    say(f"  person detector ({weights.name}, {imgsz} px)")
    say(f"    PyTorch        {pt_ms:7.0f} ms")
    best = {"runtime": "pytorch", "ms": pt_ms}
    try:
        import openvino  # noqa: F401
        build = out / "build"
        build.mkdir(parents=True, exist_ok=True)
        shutil.copy(weights, build / weights.name)
        exported = Path(YOLO(str(build / weights.name)).export(
            format="openvino", imgsz=imgsz, half=False, dynamic=False, verbose=False))
        model = YOLO(str(exported), task="pose")
        got = [boxes(model)(f).boxes.xyxy.tolist() for f in frames]
        agree = float(np.mean([same_people(r, g) for r, g in zip(ref, got)]))
        ms = timed(boxes(model), frames)
        ok = agree >= 0.9
        say(f"    OpenVINO       {ms:7.0f} ms   same people found: {agree:.0%}"
            + ("" if ok else "  (too different: not used)"))
        if ok and ms <= FASTER * pt_ms:
            dest = out / f"{weights.stem}_{imgsz}_openvino_model"   # ultralytics knows the format by this ending
            shutil.rmtree(dest, ignore_errors=True)
            shutil.move(str(exported), dest)
            boxes(YOLO(str(dest), task="pose"))(frames[0])   # loads from where the monitor will
            best = {"runtime": "openvino", "ms": ms, "path": dest.relative_to(ROOT).as_posix(),
                    "agreement": round(agree, 3)}
        shutil.rmtree(build, ignore_errors=True)
    except Exception as e:
        say(f"    OpenVINO       unavailable ({type(e).__name__}: {str(e)[:100]})")
    say(f"    -> {best['runtime']}")
    best.update(weights=weights.name, imgsz=imgsz, pytorch_ms=round(pt_ms, 1), ms=round(best["ms"], 1))
    return best


def optimise_activity(frames, out: Path) -> dict:
    import torch
    from src.action_recognition.clip_encoder import CLIP_LEN, ClipEncoder, prepare_frame
    enc = ClipEncoder("s3d", device="cpu")
    enc._compiled = None                                  # time PyTorch itself
    m = enc.model

    class Both(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            pooled = self.m.avgpool(self.m.features(x))
            return self.m.classifier(pooled).mean((2, 3, 4)), pooled.mean((2, 3, 4))

    both = Both(m).eval()
    # Real clips: 16 consecutive prepared frames, normalised as clip_encoder does.
    clips = []
    for i in range(0, max(1, len(frames) - CLIP_LEN + 1), max(1, len(frames) // 3)):
        seq = [prepare_frame(f) for f in (frames * CLIP_LEN)[i:i + CLIP_LEN]]
        rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in seq])
        x = torch.from_numpy(rgb).float().div_(255).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        clips.append(((x - enc._mean) / enc._std).numpy())
        if len(clips) == 3:
            break

    with torch.inference_mode():
        refs = [tuple(t.numpy() for t in both(torch.from_numpy(x))) for x in clips]
        pt_ms = timed(lambda x: both(torch.from_numpy(x)), clips, repeat=2)
    say("  activity model (S3D)")
    say(f"    PyTorch        {pt_ms:7.0f} ms")
    best = {"runtime": "pytorch", "ms": pt_ms}

    onnx = out / "s3d.onnx"
    try:
        with torch.inference_mode():
            torch.onnx.export(both, torch.from_numpy(clips[0]), str(onnx), input_names=["clip"],
                              output_names=["logits", "feat"], opset_version=17, dynamo=False)
    except Exception as e:
        say(f"    (could not export: {type(e).__name__}: {str(e)[:100]})")
        return {"s3d": dict(best, pytorch_ms=round(pt_ms, 1))}

    for runtime, label in (("onnxruntime", "ONNX Runtime"), ("openvino", "OpenVINO")):
        try:
            run = cm.load_activity(onnx, runtime)
            run(clips[0])                                 # compile / warm up
            t = time.perf_counter()
            got = run(clips[0])
            first_ms = (time.perf_counter() - t) * 1000
            diff = max(float(np.abs(got[0] - refs[0][0]).max()), float(np.abs(got[1] - refs[0][1]).max()))
            # Far slower than PyTorch (OpenVINO's 3D convolutions on ARM:
            # 30 s a look): no need to time it further.
            ms = first_ms if first_ms > 3 * pt_ms else timed(run, clips, repeat=2)
            ok = diff < 1e-3
            say(f"    {label:14} {ms:7.0f} ms   largest score difference {diff:.1e}"
                + ("" if ok else "  (too different: not used)"))
            if ok and ms <= FASTER * best["ms"]:
                best = {"runtime": runtime, "ms": ms, "path": onnx.relative_to(ROOT).as_posix()}
        except Exception as e:
            say(f"    {label:14} unavailable ({type(e).__name__}: {str(e)[:100]})")
    if best["runtime"] == "pytorch":
        onnx.unlink(missing_ok=True)
    say(f"    -> {best['runtime']}")
    best.update(pytorch_ms=round(pt_ms, 1), ms=round(best["ms"], 1))
    return {"s3d": best}


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--remove", action="store_true", help="delete the compiled models")
    a = ap.parse_args()
    if a.remove:
        shutil.rmtree(cm.DIR, ignore_errors=True)
        say("Compiled models removed; the monitor uses PyTorch.")
        return 0

    say(f"Optimising for: {summary(settings.device)}")
    if str(settings.device).startswith("cuda"):
        if is_jetson():
            import subprocess
            return subprocess.call([sys.executable, str(ROOT / "scripts" / "build_trt_engines.py")])
        say("NVIDIA GPU: the models already run with CUDA; nothing to compile.")
        return 0

    import torch
    out = cm.DIR
    out.mkdir(parents=True, exist_ok=True)
    frames = demo_frames()
    torch.set_num_threads(cm.threads())
    say(f"  {len(frames)} frames from the demo clips; {cm.threads()} threads per model\n")
    manifest = {
        "made": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": describe(),
        "torch": torch.__version__,
        "detector": optimise_detector(frames, out),
        "activity": optimise_activity(frames, out),
    }
    cm.MANIFEST.write_text(json.dumps(manifest, indent=2))
    d, s = manifest["detector"], manifest["activity"]["s3d"]
    say(f"\nSaved {cm.MANIFEST.relative_to(ROOT)}:")
    say(f"  person detector  {d['runtime']:12} {d['ms']:.0f} ms per frame (PyTorch {d['pytorch_ms']:.0f})")
    say(f"  activity model   {s['runtime']:12} {s['ms']:.0f} ms per look  (PyTorch {s['pytorch_ms']:.0f})")
    say("Used from the next start of the monitor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
