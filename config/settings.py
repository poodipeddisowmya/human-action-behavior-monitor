"""Central runtime configuration.

Loads from environment (.env) and merges with model_config.yaml. Every module
imports `settings` from here so no path or hyperparameter is hard-coded elsewhere.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Keep torchvision's pretrained weights inside the project.
#
# By default torch caches downloads in ~/.cache/torch, which makes a fresh
# machine depend on an internet connection and on whatever version the CDN
# serves that day. Pointing TORCH_HOME at models/torch means the bundled
# s3d checkpoint is found immediately and the app runs offline and identically
# on any machine. torch reads this lazily (at weight-load time, not import),
# so setting it here is early enough as long as `config` is imported first —
# which every module does.
#
# setdefault, not assignment: an operator who has deliberately set TORCH_HOME
# keeps their choice.
os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch"))

# The monitor never draws charts, but ultralytics imports matplotlib, which
# fails outright if MPLBACKEND names a backend this environment lacks.
# Jupyter sets exactly that (module://matplotlib_inline...) for every program
# a notebook launches — so on Kaggle the server died at import with
# "'module://matplotlib_inline.backend_inline' is not a valid value for
# backend". Assignment, not setdefault: the inherited value is the problem.
os.environ["MPLBACKEND"] = "Agg"


class ModelConfig(BaseModel):
    detector_weights: str
    detector_conf: float
    detector_iou: float
    tracker: str
    pose_model: str
    action_classifier: str
    action_window: int
    action_stride: int
    action_labels: list[str]
    # Deep action-recognition backend: 's3d' (fast) or 'videomae' (slow).
    deep_backend: str = "s3d"
    # YOLO input resolution — sets the video FPS ceiling.
    detector_imgsz: int = 416
    # Minimum gap between deep inferences. Declared here because pydantic
    # drops undeclared keys: without this field the value in
    # model_config.yaml was silently ignored and the code default applied.
    deep_min_interval_s: float = 2.5
    # Plain-English caption under the video (src/description). Turning it
    # off removes only the place recogniser; captions from the action labels
    # still appear.
    scene_description: bool = True
    # GPU only: half-precision person detector (about 2x faster on T4-class
    # cards, same boxes to within a pixel). The activity model stays FP32
    # because its confidence thresholds were calibrated in FP32.
    detector_half: bool = False
    # Used when detector_weights is not on disk and cannot be downloaded
    # (an offline GPU box asked for a larger model than is bundled).
    detector_fallback_weights: str = "yolov8n-pose.pt"
    # A posture change must persist this long before the on-screen label
    # switches (src/temporal/stabilizer.py). 0 disables. Display only.
    label_min_dwell_s: float = 0.8
    # Detailed scene-aware subtitles from a vision-language model
    # (src/description/narrator.py). GPU only: ~100 s per sentence on a CPU.
    vlm_captions: bool = False
    # Detector input size for HD video (frame width >= detector_hd_min_width);
    # 0 = always detector_imgsz. GPU only: 960 px tracked 5x more people in
    # a 768x576 street scene on a T4, but upscaling a 320x240 clip to 960 px
    # only added noise (one clip read "falling"), hence the width gate.
    detector_imgsz_hd: int = 0
    detector_hd_min_width: int = 720


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).split("#")[0].strip().lower() in ("1", "true", "yes", "on")


def _resolve_device(choice: str) -> str:
    """'auto' (the default) uses an NVIDIA GPU when one is usable, else CPU.

    The same install then runs on this laptop and on a GPU machine with no
    edits. DEVICE=cpu / DEVICE=cuda in .env still forces a choice.
    """
    choice = (choice or "auto").split("#")[0].strip().lower()
    if choice not in ("", "auto") and not choice.startswith("cuda"):
        return choice
    try:
        import torch
        if torch.cuda.is_available():
            # is_available() only says a driver answered. A PyTorch build
            # without kernels for this GPU (e.g. a new wheel on an old
            # Pascal card) still reports True and then fails on the first
            # real op, deep inside the pipeline thread. Run one op here so
            # that case falls back to the CPU with a clear message instead.
            dev = choice if choice.startswith("cuda") else "cuda"
            (torch.ones(8, device=dev) * 2).sum().item()
            torch.backends.cudnn.benchmark = True       # fixed input sizes
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return dev
    except Exception as e:
        print(f"*** GPU present but unusable ({e}); running on the CPU.", flush=True)
        return "cpu"
    if choice.startswith("cuda"):
        print("*** DEVICE=cuda requested but no usable GPU; running on the CPU.", flush=True)
    return "cpu"


class Settings(BaseModel):
    root: Path = ROOT
    models_dir: Path = ROOT / "models"
    data_dir: Path = ROOT / "data"
    logs_dir: Path = ROOT / "logs"

    api_host: str = Field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    ui_origin: str = Field(default_factory=lambda: os.getenv("UI_ORIGIN", "http://localhost:5173"))
    database_url: str = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            f"sqlite+aiosqlite:///{(ROOT / 'logs' / 'monitor.db').as_posix()}",
        )
    )

    device: str = Field(default_factory=lambda: _resolve_device(os.getenv("DEVICE", "auto")))
    target_fps: int = Field(default_factory=lambda: int(os.getenv("TARGET_FPS", "15")))
    confidence_threshold: float = Field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    )

    # Shared secret for a server reachable from outside this machine (a
    # Kaggle tunnel, a cloud VM). Empty = open, which is right for localhost.
    # Open the dashboard once as  https://host/?token=<ACCESS_TOKEN>  and a
    # cookie carries it from then on (the <img> video stream and the
    # WebSocket cannot send headers, so a header-only scheme would not work).
    access_token: str = Field(default_factory=lambda: os.getenv("ACCESS_TOKEN", "").strip())

    # Video stream to the browser. Frames are encoded once and shared by
    # every viewer; the width cap keeps a 1080p clip watchable over a tunnel.
    stream_fps: int = Field(default_factory=lambda: int(os.getenv("STREAM_FPS", "20")))
    stream_max_width: int = Field(default_factory=lambda: int(os.getenv("STREAM_MAX_WIDTH", "1280")))
    stream_jpeg_quality: int = Field(default_factory=lambda: int(os.getenv("STREAM_JPEG_QUALITY", "80")))

    # Probe local USB cameras for the Source list. Off on headless servers
    # (Kaggle, Docker), where probing only costs seconds and finds nothing;
    # the browser's own webcam is offered there instead.
    probe_cameras: bool = Field(default_factory=lambda: _env_bool(
        "PROBE_CAMERAS", default=os.name == "nt" or bool(os.getenv("DISPLAY"))))

    model: ModelConfig


@lru_cache
def _load() -> Settings:
    with (ROOT / "config" / "model_config.yaml").open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    s = Settings(model=ModelConfig(**raw["model"]))
    if s.device.startswith("cuda"):
        # On a GPU the speed/responsiveness trade-offs made for the CPU no
        # longer apply: look at the activity far more often (labels confirm
        # in about a second instead of several) and run the person detector
        # at full resolution, which finds smaller and more distant people.
        gpu = raw.get("gpu", {}) or {}
        s.model.deep_min_interval_s = float(gpu.get("deep_min_interval_s", 0.4))
        s.model.detector_imgsz = int(gpu.get("detector_imgsz", 640))
        s.model.detector_half = bool(gpu.get("detector_half", True))
        s.model.detector_imgsz_hd = int(gpu.get("detector_imgsz_hd", 0))
        s.model.vlm_captions = bool(gpu.get("vlm_captions", False))
        if gpu.get("deep_backend"):
            s.model.deep_backend = str(gpu["deep_backend"]).lower()
        if gpu.get("detector_weights"):
            s.model.detector_fallback_weights = s.model.detector_weights
            s.model.detector_weights = str(gpu["detector_weights"])
    # Operator overrides without editing YAML (Docker / Kaggle).
    if os.getenv("DETECTOR_WEIGHTS"):
        s.model.detector_weights = os.environ["DETECTOR_WEIGHTS"].strip()
    if os.getenv("DETECTOR_IMGSZ"):
        s.model.detector_imgsz = int(os.environ["DETECTOR_IMGSZ"])
    if os.getenv("DETECTOR_IMGSZ_HD"):
        s.model.detector_imgsz_hd = int(os.environ["DETECTOR_IMGSZ_HD"])
    if os.getenv("DETECTOR_HALF"):
        s.model.detector_half = _env_bool("DETECTOR_HALF")
    if os.getenv("DEEP_MIN_INTERVAL_S"):
        s.model.deep_min_interval_s = float(os.environ["DEEP_MIN_INTERVAL_S"])
    if os.getenv("DEEP_BACKEND"):
        s.model.deep_backend = os.environ["DEEP_BACKEND"].strip().lower()
    if os.getenv("VLM_CAPTIONS"):
        s.model.vlm_captions = _env_bool("VLM_CAPTIONS")
    if os.getenv("LABEL_MIN_DWELL_S"):
        s.model.label_min_dwell_s = float(os.environ["LABEL_MIN_DWELL_S"])
    return s


settings = _load()
