"""Models compiled for a CPU (src/utils/compiled_models.py,
scripts/optimise_for_device.py): when they are used, and that a broken one
never stops the monitor."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.utils import compiled_models as cm

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cpu_machine(tmp_path, monkeypatch):
    """A CPU machine whose project root is tmp_path, with a compiled detector."""
    monkeypatch.setattr(cm.settings, "device", "cpu")
    monkeypatch.setattr(cm.settings, "root", tmp_path)
    monkeypatch.setattr(cm, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.delenv("CPU_RUNTIME", raising=False)
    (tmp_path / "det_416_openvino_model").mkdir()
    (tmp_path / "s3d.onnx").write_bytes(b"x")
    cm.MANIFEST.write_text(json.dumps({
        "detector": {"runtime": "openvino", "weights": "yolov8n-pose.pt", "imgsz": 416,
                     "path": "det_416_openvino_model"},
        "activity": {"s3d": {"runtime": "pytorch"}},
    }))
    return tmp_path


def test_detector_used_only_when_it_matches(cpu_machine, monkeypatch):
    assert cm.detector_for("yolov8n-pose.pt", 416).endswith("det_416_openvino_model")
    assert cm.detector_for("yolov8s-pose.pt", 416) is None        # other weights
    assert cm.detector_for("yolov8n-pose.pt", 640) is None        # other input size
    monkeypatch.setenv("CPU_RUNTIME", "pytorch")
    assert cm.detector_for("yolov8n-pose.pt", 416) is None        # switched off
    monkeypatch.delenv("CPU_RUNTIME")
    monkeypatch.setattr(cm.settings, "device", "cuda")
    assert cm.detector_for("yolov8n-pose.pt", 416) is None        # a GPU does its own


def test_activity_model_pytorch_or_missing_means_none(cpu_machine):
    assert cm.activity_model("s3d") is None                       # PyTorch was fastest
    assert cm.activity_model("videomae") is None                  # never compiled
    cm.MANIFEST.write_text("not json")
    assert cm.manifest() == {} and cm.detector_for("yolov8n-pose.pt", 416) is None


def test_broken_compiled_detector_falls_back_to_pytorch(monkeypatch, capsys):
    """A compiled model that cannot load must not blank the video: the
    tracker checks it once and runs the PyTorch weights instead."""
    import src.utils.compiled_models as mod
    from src.tracking import person_tracker as pt
    monkeypatch.setattr(mod, "detector_for", lambda w, s: str(ROOT / "README.md"))
    monkeypatch.setattr(pt.settings.model, "detector_engine", False)   # a Jetson's own path
    tracker = pt.PersonTracker(weights="yolov8n-pose.pt", device="cpu")
    assert tracker.weights_name == "yolov8n-pose.pt"
    assert "compiled model unusable" in capsys.readouterr().out
    assert tracker.detect(np.zeros((240, 320, 3), np.uint8)) == []


def test_same_people_matching():
    spec = importlib.util.spec_from_file_location("opt", ROOT / "scripts" / "optimise_for_device.py")
    opt = importlib.util.module_from_spec(spec)
    sys.modules["opt"] = opt
    spec.loader.exec_module(opt)
    a = [[0, 0, 100, 200], [300, 0, 400, 200]]
    assert opt.same_people(a, list(reversed(a))) == 1.0             # order does not matter
    assert opt.same_people(a, [[2, 3, 101, 198]]) == 0.5            # one of two found
    assert opt.same_people([], []) == 1.0


def test_hardware_summary():
    from src.utils.hardware import describe, summary
    d = describe()
    assert d["cores"] >= 1 and d["arch"]
    assert "cores" in summary("cpu") and "the CPU" in summary("cpu")
