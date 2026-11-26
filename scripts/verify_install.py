"""Preflight check — run this first on a new machine.

Answers "will this work here?" with a list of specific, fixable statements
instead of a stack trace three layers into a library. Every check prints what
it looked for and what it found.

    python scripts/verify_install.py
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, BAD, WARN = "  OK  ", " FAIL ", " WARN "
failures: list[str] = []
warnings: list[str] = []


def check(label: str, ok: bool, detail: str = "", fix: str = "", warn_only: bool = False) -> None:
    if ok:
        print(f"[{OK}] {label}" + (f"  — {detail}" if detail else ""))
        return
    tag = WARN if warn_only else BAD
    print(f"[{tag}] {label}" + (f"  — {detail}" if detail else ""))
    if fix:
        print(f"         fix: {fix}")
    (warnings if warn_only else failures).append(label)


def main() -> None:
    print("\nHuman Action & Behavior Monitor — install check\n" + "-" * 52)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.utils.hardware import summary
        print(f"  machine: {summary()}\n")
    except Exception:
        pass

    # --- Python version ---------------------------------------------------
    # Upper bound matters as much as the lower one: numpy 1.26.4 ships no
    # wheels for 3.13+, so a newer Python turns `pip install` into a source
    # build that dies in a compiler error nobody can read. Someone installing
    # Python today gets the newest version by default, so this is the most
    # likely way setup fails on a fresh machine.
    v = sys.version_info
    check(
        "Python 3.10-3.12",
        (3, 10) <= (v.major, v.minor) <= (3, 12),
        f"found {v.major}.{v.minor}.{v.micro}",
        "install Python 3.11 from python.org, delete the .venv folder, "
        "and run setup.bat again",
    )

    # --- Large files (Git LFS) ---------------------------------------------
    # Model weights, demo clips and the comparison videos are stored with Git
    # LFS. A clone made without Git LFS installed, or GitHub's "Download ZIP",
    # leaves small text pointers in their place, and everything fails later
    # with unreadable errors (a "corrupt" model file, a video that will not
    # open). Catch it here, with the fix.
    root = Path(__file__).resolve().parents[1]
    pointers = []
    for pattern in ("models/*.pt", "models/face/*.onnx", "data/samples/*.mp4",
                    "data/samples/*.avi", "data/showcase/*.mp4", "data/showcase/*/*.mp4"):
        for f in root.glob(pattern):
            try:
                with open(f, "rb") as fh:
                    if fh.read(40).startswith(b"version https://git-lfs"):
                        pointers.append(f.relative_to(root).as_posix())
            except OSError:
                pass
    check("large files downloaded (Git LFS)", not pointers,
          f"{len(pointers)} file(s) are LFS pointers, e.g. {pointers[0]}" if pointers else "",
          "install Git LFS (https://git-lfs.com), then in this folder run: "
          "git lfs install  and  git lfs pull")

    # --- Third-party packages --------------------------------------------
    # Imported by module name, since the pip name and import name differ for
    # several of these (opencv-python -> cv2, pyyaml -> yaml).
    for mod, pip_name in [
        ("cv2", "opencv-python"), ("torch", "torch"), ("torchvision", "torchvision"),
        ("ultralytics", "ultralytics"), ("lap", "lapx"), ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn[standard]"), ("sqlalchemy", "sqlalchemy"),
        ("aiosqlite", "aiosqlite"), ("yaml", "pyyaml"), ("loguru", "loguru"),
        ("multipart", "python-multipart"),
    ]:
        try:
            importlib.import_module(mod)
            check(f"package {pip_name}", True)
        except Exception as e:
            check(f"package {pip_name}", False, str(e)[:60],
                  "pip install -r requirements.txt")

    if failures:
        print("\nStopping: install the packages above first.\n")
        sys.exit(1)

    # --- Config + weights -------------------------------------------------
    from config import settings
    from src.tracking.person_tracker import resolve_weights

    wname = settings.model.detector_weights
    check("detector is a pose model", "pose" in wname, wname,
          "set detector_weights to a '-pose' model in config/model_config.yaml")

    resolved = resolve_weights(wname)
    bundled = Path(resolved).exists()
    check("YOLO pose weights bundled", bundled,
          resolved if bundled else f"{wname} not found in models/",
          "re-download: python scripts/download_models.py",
          warn_only=True)

    s3d = settings.root / "models" / "torch" / "hub" / "checkpoints" / "s3d-d76dad2f.pth"
    check("s3d action weights bundled", s3d.exists(),
          f"{s3d.stat().st_size // (1024*1024)} MB" if s3d.exists()
          else "missing — will download on first run (needs internet)",
          "python scripts/download_models.py", warn_only=True)

    # --- Sample clips -----------------------------------------------------
    clips = list((settings.data_dir / "samples").glob("*.avi")) + \
            list((settings.data_dir / "samples").glob("*.mp4"))
    check("demo clips present", bool(clips), f"{len(clips)} clip(s)",
          "add a video to data/samples/, or pass --source <file>")

    # --- Built UI ---------------------------------------------------------
    ui = settings.root / "ui" / "dist" / "index.html"
    check("built UI present", ui.exists(),
          "ui/dist/index.html" if ui.exists() else "missing — API will run without a UI",
          "cd ui && npm install && npm run build")

    # --- Writable dirs ----------------------------------------------------
    try:
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        probe = settings.logs_dir / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        check("logs/ writable", True)
    except Exception as e:
        check("logs/ writable", False, str(e)[:60],
              "check folder permissions, or move the project out of a synced folder")

    # --- The real test: load the model and run one frame ------------------
    print("\nLoading models (first run may take a moment)...")
    try:
        import numpy as np
        from src.tracking import PersonTracker

        tracker = PersonTracker()
        tracker.update(np.zeros((480, 640, 3), dtype=np.uint8))
        check("model loads and runs a frame", True)
    except Exception as e:
        check("model loads and runs a frame", False, str(e)[:100],
              "see the traceback above; usually a missing weight file or a bad install")

    # --- Device and speed-ups ---------------------------------------------
    gpu = str(settings.device).startswith("cuda")
    check("runs on", True, "the NVIDIA GPU" if gpu else "the CPU (no NVIDIA GPU found)")
    if not gpu:
        from src.utils import compiled_models as cm
        man = cm.manifest()
        if man:
            d, a = man.get("detector") or {}, (man.get("activity") or {}).get("s3d") or {}
            check("compiled for this CPU", True,
                  f"detector {d.get('runtime')} {d.get('ms', 0):.0f} ms, "
                  f"activity model {a.get('runtime')} {a.get('ms', 0):.0f} ms")
        else:
            check("compiled for this CPU", False, "not yet: PyTorch models, several times slower",
                  "python scripts/optimise_for_device.py", warn_only=True)

    # --- Trained activities (docs/TRAINING.md): optional ----------------
    try:
        from src.action_recognition.custom_activities import ActivityHead, model_paths
        backend = settings.model.deep_backend
        npz, meta = model_paths(backend)
        others = [b for b in ("s3d", "videomae") if b != backend and model_paths(b)[0].is_file()]
        if npz.is_file():
            head = ActivityHead.load(npz, meta)
            check("trained activities", True, f"{', '.join(head.activities)} ({backend})")
        elif others:
            check("trained activities", False, f"trained for {others[0]}, but this machine uses {backend}",
                  f"python scripts/train_activities.py --backend {backend}", warn_only=True)
    except Exception as e:
        check("trained activities", False, str(e)[:100],
              "retrain: python scripts/train_activities.py", warn_only=True)

    # --- Verdict ----------------------------------------------------------
    print("-" * 52)
    if failures:
        print(f"\n{len(failures)} problem(s): " + ", ".join(failures))
        print("Fix these, then re-run this script.\n")
        sys.exit(1)
    if warnings:
        print(f"\n{len(warnings)} warning(s): " + ", ".join(warnings))
        print("These will still work if the machine has internet on first run.")
    start = "run.bat" if os.name == "nt" else "./run.sh"
    print(f"\nAll good. Start it with:\n\n    {start}\n")
    print("Then open http://localhost:8000\n")


if __name__ == "__main__":
    main()
