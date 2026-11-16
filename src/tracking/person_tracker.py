"""Multi-object tracker + whole-frame pose in one pass.

Uses Ultralytics' native ByteTrack integration on YOLOv8n-pose. A single
forward pass yields, per detected person:
    * stable track ID (survives short occlusions)
    * bounding box in original frame coords
    * 17 COCO keypoints (x, y, confidence)

Compared to detector→crop→MediaPipe, this is:
    * more accurate on distant / small people (whole-frame context)
    * ~30% faster on multi-person scenes (no per-person crop pass)
    * one model to manage, not three
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# config first: it sets MPLBACKEND, which ultralytics' matplotlib import needs.
from config import settings
from ultralytics import YOLO  # noqa: E402

PERSON_CLASS_ID = 0


def resolve_weights(name: str) -> str:
    """Find a weights file inside the project, wherever the server is launched from.

    Ultralytics resolves a bare filename against the *current working
    directory* and, failing that, downloads it from the internet. Both are
    wrong for a shipped copy: starting the server from another directory would
    silently pull a fresh download (or fail outright on an offline machine),
    and the downloaded file is not necessarily the version this was tuned
    against. Looking inside the project first makes the bundled weights
    authoritative regardless of cwd.

    Falls back to the bare name — and therefore to Ultralytics' downloader —
    only when the file genuinely is not bundled.
    """
    if Path(name).is_absolute() and Path(name).exists():
        return name
    for candidate in (settings.models_dir / name, settings.root / name):
        if candidate.exists():
            return str(candidate)
    return name


def fetch_weights(name: str, fallback: str | None = None) -> str:
    """Local path for `name`, downloading it into models/ if needed.

    A GPU deployment may ask for a larger detector than the one bundled.
    Downloading into models/ (not the cwd, which is where Ultralytics would
    put it) means it is fetched once and found on every later start. If the
    download fails — an offline site — the bundled fallback is used and the
    reason is printed, so the monitor still starts.
    """
    local = resolve_weights(name)
    if Path(local).exists():
        return local
    try:
        from ultralytics.utils.downloads import attempt_download_asset
        target = settings.models_dir / Path(name).name
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        got = attempt_download_asset(str(target))
        if Path(got).exists():
            print(f"[detector] downloaded {name} -> {got}", flush=True)
            return str(got)
    except Exception as e:
        print(f"[detector] could not download {name}: {e}", flush=True)
    if fallback and fallback != name:
        print(f"*** detector '{name}' unavailable; using bundled '{fallback}'.", flush=True)
        return resolve_weights(fallback)
    return local


@dataclass
class Track:
    person_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    keypoints: np.ndarray | None = field(default=None)   # (17, 3) COCO order, or None


class PersonTracker:
    def __init__(self, weights: str | None = None, device: str | None = None):
        chosen = weights or settings.model.detector_weights
        path = fetch_weights(chosen, getattr(settings.model, "detector_fallback_weights", None))
        chosen = Path(path).name
        if getattr(settings.model, "detector_engine", False):
            path = self._engine_for(path)
            chosen = Path(path).name
        self.weights_name = chosen
        self._model = YOLO(path, task="pose" if "pose" in chosen else None)
        self._device = device or settings.device
        self._half = str(self._device).startswith("cuda") and getattr(
            settings.model, "detector_half", False)
        self._tracker_cfg = settings.model.tracker
        self._conf = settings.model.detector_conf
        # HD-size state (see _imgsz). Start at the base size: a close-up's
        # first frame must not be the one that shows phantom boxes.
        self._use_hd = False
        self._tallest = 1.0         # tallest person / frame height, last frame
        self._pose_capable = "pose" in chosen
        if not self._pose_capable:
            # Loud, because the failure mode is silent and severe: without
            # keypoints every pose rule (sitting / drinking / waving /
            # clapping / reading / push-up / crawling / bending) is dead and
            # the system degrades to motion-only labels with no error anywhere.
            # This shipped once by accident and cost a lot of debugging.
            print(
                f"\n*** WARNING: detector_weights='{chosen}' has no pose head.\n"
                f"***   All keypoint-based action rules are DISABLED.\n"
                f"***   Only motion labels (standing/walking/running) will appear.\n"
                f"***   Set detector_weights to a '-pose' model in "
                f"config/model_config.yaml.\n",
                flush=True,
            )

    @staticmethod
    def _engine_for(weights: str) -> str:
        """The TensorRT engine built from `weights`, or `weights` if none.

        Built once per device by scripts/build_trt_engines.py (an engine is
        specific to the GPU and TensorRT version that built it, so it is not
        shipped). It must cover the largest input size the tracker asks for.
        """
        engine = Path(weights).with_suffix(".engine")
        if not engine.exists():
            print(f"[detector] no TensorRT engine at {engine}; using {Path(weights).name} "
                  f"(build it: python scripts/build_trt_engines.py)", flush=True)
            return weights
        try:
            import json
            import struct
            with engine.open("rb") as f:          # ultralytics: <len><json meta><engine>
                meta = json.loads(f.read(struct.unpack("<i", f.read(4))[0]).decode())
            built = max(meta.get("imgsz") or [0])
            need = max(getattr(settings.model, "detector_imgsz", 640),
                       getattr(settings.model, "detector_imgsz_hd", 0))
            if built < need:
                print(f"[detector] {engine.name} covers {built} px but {need} px is "
                      f"configured; using {Path(weights).name}", flush=True)
                return weights
        except Exception as e:
            print(f"[detector] unreadable engine {engine.name} ({e}); "
                  f"using {Path(weights).name}", flush=True)
            return weights
        print(f"[detector] TensorRT engine {engine.name}", flush=True)
        return str(engine)

    def warmup(self) -> None:
        """One throwaway inference, so the first real frame is not the slow
        one (CUDA context, cuDNN autotune and FP16 fusion all happen here).
        Uses predict, not track, so no tracker state is created. Both input
        sizes are warmed: each is a separate autotune on a GPU."""
        sizes = {getattr(settings.model, "detector_imgsz", 640),
                 getattr(settings.model, "detector_imgsz_hd", 0)} - {0}
        for sz in sorted(sizes):
            self._model.predict(np.zeros((sz, sz, 3), np.uint8), imgsz=sz,
                                device=self._device, half=self._half, verbose=False)

    # The larger HD input size is used only while the people in view are
    # small: it is for finding distant people, and on a close-up it hurts.
    # Measured (yolov8s-pose, people per frame, 640 px vs 960 px):
    #   street scene, ~6 small people     1.33 -> 4.00   (real people found)
    #   one-person kitchen close-up        1.00 -> 2.42   (phantom boxes: slices
    #       of the same person, which the subtitle then called "three people")
    # Hysteresis (two thresholds) stops it flipping every frame. A phantom is
    # itself a large box, so the rule corrects itself.
    _HD_ON_MAX_H = 0.35      # use HD while the tallest person is under 35% of the frame
    _HD_OFF_MAX_H = 0.45     # ...and drop back once anyone is over 45%

    def _imgsz(self, image: np.ndarray) -> int:
        """Detector input size for this frame."""
        base = getattr(settings.model, "detector_imgsz", 640)
        hd = getattr(settings.model, "detector_imgsz_hd", 0)
        if not hd or image.shape[1] < getattr(settings.model, "detector_hd_min_width", 720):
            return base
        if self._use_hd and self._tallest > self._HD_OFF_MAX_H:
            self._use_hd = False
        elif not self._use_hd and self._tallest < self._HD_ON_MAX_H:
            self._use_hd = True
        return hd if self._use_hd else base

    def update(self, image: np.ndarray) -> list[Track]:
        kwargs = dict(
            conf=self._conf,
            tracker=self._tracker_cfg,
            persist=True,
            device=self._device,
            verbose=False,
            # Input resolution sets the per-frame cost and therefore the video
            # FPS ceiling. Measured: 640 -> 77ms, 416 -> 38ms (2x faster).
            imgsz=self._imgsz(image),
            half=self._half,
        )
        # Pose models are single-class (person); detection models take classes=.
        if not self._pose_capable:
            kwargs["classes"] = [PERSON_CLASS_ID]

        result = self._model.track(image, **kwargs)[0]

        out: list[Track] = []
        boxes = result.boxes.xyxy if result.boxes is not None else None
        self._tallest = (float((boxes[:, 3] - boxes[:, 1]).max()) / image.shape[0]
                         if boxes is not None and len(boxes) else 0.0)
        if result.boxes is None or result.boxes.id is None:
            return out

        ids = result.boxes.id.int().tolist()
        kps_all = None
        if self._pose_capable and result.keypoints is not None and result.keypoints.data is not None:
            # shape (N, 17, 3) — x, y, confidence
            kps_all = result.keypoints.data.cpu().numpy()

        for idx, (tid, box) in enumerate(zip(ids, result.boxes)):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            kp = kps_all[idx] if kps_all is not None and idx < len(kps_all) else None
            out.append(Track(
                person_id=tid,
                xyxy=(x1, y1, x2, y2),
                confidence=float(box.conf[0]),
                keypoints=kp,
            ))
        return out
