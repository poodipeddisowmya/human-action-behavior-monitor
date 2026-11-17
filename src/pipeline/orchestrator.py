"""End-to-end pipeline.

     VideoSource → PersonTracker(YOLOv8-pose + ByteTrack) → ActionClassifier
                                                          ↓
                          TemporalSmoother → EventGenerator → callbacks
"""
from __future__ import annotations

import time
from datetime import datetime
from threading import Event, Lock, Thread
from typing import Callable

import numpy as np

from config import settings
from src.action_recognition import ActionClassifier
from src.description import SceneDescriber
from src.input import VideoSource
from src.input.video_source import is_live
from src.temporal import EventGenerator, TemporalSmoother
from src.temporal.event_generator import ActionEvent
from src.temporal.stabilizer import LabelStabilizer
from src.tracking import PersonTracker, Track
from src.utils import draw_overlays, get_logger

log = get_logger(__name__)

# Cap on adaptive frame skipping. Beyond this, boxes lag far enough behind
# the person that the overlay looks wrong; better to drop playback speed.
_MAX_SKIP = 8

FrameCallback = Callable[[np.ndarray, list[dict]], None]
EventCallback = Callable[[ActionEvent], None]


class Pipeline:
    def __init__(
        self,
        source: VideoSource,
        on_frame: FrameCallback | None = None,
        on_event: EventCallback | None = None,
    ):
        self._source = source
        self._on_frame = on_frame
        self._on_event = on_event or (lambda e: log.info(f"event: {e.to_dict()}"))

        self._tracker = PersonTracker()   # detection + pose + tracking in one pass
        self._classifier = ActionClassifier()
        # Smoothing: small window kills flicker; low floor because the deep
        # classifier already applies its own confidence bar upstream.
        self._smoother = TemporalSmoother(window=3, min_confidence=0.4)
        # What is SHOWN: posture changes must persist before the display
        # switches; activities show at once (see stabilizer.py).
        self._stabilizer = LabelStabilizer(settings.model.label_min_dwell_s)
        self._events = EventGenerator(on_event=self._on_event)
        # Plain-English caption of who is doing what, and where. Additive:
        # it reads the labels decided above and never feeds back into them.
        narrator = None
        if getattr(settings.model, "vlm_captions", False):
            # Detailed scene-aware narration (GPU); see src/description/narrator.py.
            from src.description.narrator import SceneNarrator
            narrator = SceneNarrator(settings.device)
        self._describer = SceneDescriber(
            scene=getattr(settings.model, "scene_description", True), narrator=narrator)
        self._caption: dict = self._describer.current

        self._stop = Event()
        self._thread: Thread | None = None
        self._last_ids: set[int] = set()
        # Display/detection decoupling (see _detect_loop).
        self._frame_lock = Lock()
        self._pending_frame: np.ndarray | None = None
        # Video presentation time of _pending_frame. Motion speed must be
        # measured on the video clock, not the wall clock — see the note in
        # SkeletonHeuristicClassifier.update.
        self._pending_ts: float | None = None
        self._proc_ema = 0.0
        self._last_annotations: list[dict] = []
        # Track which file the source is currently playing so we can detect a
        # live source switch and clear stale per-person state.
        self._current_source = self._source_key()
        # Read by /api/metrics and /api/ready. Plain values, written by the
        # pipeline threads only, so readers need no lock.
        self.stats: dict = {
            "started_at": time.time(), "frames": 0, "playback_fps": 0.0,
            "detect_ms": 0.0, "tracked": 0, "last_frame_at": 0.0,
            "restarts": 0, "detect_errors": 0,
            "detector": getattr(self._tracker, "weights_name", settings.model.detector_weights),
        }
        self._warmup()

    def _warmup(self) -> None:
        """Pay every model's first-call cost now, before the first frame.

        The first call of each model initialises kernels (on a GPU: CUDA
        context, cuDNN autotune, FP16 fusion) and costs 1-3 s. Only the
        tracker and the activity model used to be warmed, and only on GPU;
        the object detector and CLIP paid on the clip itself. Measured on a
        T4: the first activity label took ~4.7 s on every freshly started
        pipeline, against 0.5-2.3 s once warm. Each model is warmed on its
        own so one failure cannot skip the rest.
        """
        import torch
        t0 = time.time()
        dummy = np.zeros((480, 640, 3), np.uint8)
        clf = self._classifier

        def objects():
            ctx = getattr(clf, "_object_ctx", None)
            if ctx is not None and ctx.available:
                ctx._detect_relevant(dummy)

        def clip():
            from src.description.clip_shared import encode_images
            encode_images(dummy[:224, :224])

        steps = {"detector": self._tracker.warmup,
                 "activity": lambda: self._warm_activity(clf),
                 "objects": objects,
                 "clip": clip}
        done = []
        for name, step in steps.items():
            try:
                step()
                done.append(name)
            except Exception as e:
                log.warning(f"warm-up of {name} skipped: {e}")
        if str(settings.device).startswith("cuda"):
            torch.cuda.synchronize()
        log.info(f"warm-up ({', '.join(done)}) done in {time.time() - t0:.1f}s")

    @staticmethod
    def _warm_activity(clf) -> None:
        """Run the real inference path once on a black clip.

        Calling the model directly with a fresh tensor was not enough: with
        cuDNN autotuning, the production input (a permuted view) counted as
        a new shape, and the first real look still took ~4 s on a T4
        (scripts/startup_probe.py: clip submitted at 1.7 s, answer at 5.5 s).
        Its log lines are discarded: they describe a black frame.
        """
        if getattr(clf, "_model", None) is None or not hasattr(clf, "_infer"):
            return
        import contextlib
        import io
        clip = [np.zeros((224, 224, 3), np.uint8)] * 16
        with contextlib.redirect_stdout(io.StringIO()):
            clf._infer(clip)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _source_key(self):
        """Identity of the clip currently playing, if the source exposes one."""
        cur = getattr(self._source, "current", None)
        return str(cur) if cur is not None else None

    def _handle_source_switch(self) -> None:
        """Clear every piece of per-person state that assumes one video.

        Without this, clip buffers span two videos and cached predictions
        from the old source leak into the new one, so labels lag a whole
        clip behind.
        """
        log.info(f"source switched -> {self._current_source}; clearing state")
        self._events.flush()                       # close open events honestly
        for pid in list(self._last_ids):
            self._smoother.reset(pid)
            self._stabilizer.reset(pid)
        self._last_ids.clear()
        self._last_annotations = []
        self._proc_ema = 0.0
        with self._frame_lock:
            self._pending_frame = None
        if hasattr(self._classifier, "reset_all"):
            self._classifier.reset_all()
        if hasattr(self._smoother, "reset_all"):
            self._smoother.reset_all()
        self._stabilizer.reset_all()
        self._describer.reset()
        self._caption = self._describer.current

    # -------- lifecycle --------
    def start(self) -> None:
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._events.flush()
        self._source.close()

    # -------- detection worker --------
    def _detect_loop(self, target_dt: float) -> None:
        """Detect + classify on the newest available frame, forever.

        Decoupled from playback on purpose. The display loop runs at the
        source's native rate and simply reuses the most recent boxes; this
        thread refreshes them as fast as the CPU allows. Playback speed is
        therefore independent of how slow inference happens to be, which is
        what makes the video look real-time instead of like a slideshow.

        Consequence to be aware of: boxes lag the person by up to one
        detection interval (~0.1-0.4 s here). At normal walking speed that is
        not visible; for very fast motion the box trails slightly.
        """
        while not self._stop.is_set():
            with self._frame_lock:
                img = self._pending_frame
                ts = self._pending_ts
                self._pending_frame = None
            if img is None:
                time.sleep(0.005)          # nothing new yet
                continue

            t_proc = time.time()
            try:
                tracks = self._tracker.update(img)
                self._last_annotations = self._process_tracks(img, tracks, ts)
                self._caption = self._describer.update(img, self._last_annotations)
            except Exception as e:
                self.stats["detect_errors"] += 1
                log.exception(f"detection failed: {e}")
                if "out of memory" in str(e).lower():
                    self._free_gpu_memory()
                continue
            dt = time.time() - t_proc
            self._proc_ema = 0.7 * self._proc_ema + 0.3 * dt if self._proc_ema else dt
            self.stats["detect_ms"] = round(self._proc_ema * 1000, 1)
            self.stats["tracked"] = len(self._last_annotations)

    @staticmethod
    def _free_gpu_memory() -> None:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # -------- inner loop --------
    def _run(self) -> None:
        """Supervisor: keep playback running whatever a single clip does.

        A corrupt file, a camera unplugged mid-read or a decoder error used
        to end this thread, leaving the dashboard frozen on its last frame
        with nothing but a traceback in the log. Now the error is logged and
        playback resumes after a short, growing back-off. A source that ends
        normally (a one-shot file in batch mode) still ends the pipeline.
        """
        log.info("pipeline started")
        det = Thread(target=self._detect_loop, args=(0.0,), daemon=True)
        det.start()
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._play()
                if self._stop.is_set() or not hasattr(self._source, "switch_to"):
                    break                      # stopped, or a one-shot source ended
                # A swappable (live/looping) source never ends by design; if
                # it did, the clip could not be read. Wait for a new one.
                raise RuntimeError(f"source {self._source_key()} stopped producing frames")
            except Exception as e:
                self.stats["restarts"] += 1
                log.exception(f"playback failed ({e}); restarting in {backoff:.0f}s")
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)
        self._events.flush()
        log.info("pipeline stopped")

    def _play(self) -> None:
        target_dt = 1.0 / max(1, int(self._source.fps))
        # Rolling throughput measurement — reported every 5s so "is it fast
        # enough" is answered with a number instead of an impression.
        fps_t0, fps_frames = time.time(), 0
        for frame in self._source.frames():
            if self._stop.is_set():
                break
            t0 = time.time()

            # Detect a live source switch and wipe state that assumed the
            # previous video.
            key = self._source_key()
            if key != self._current_source:
                self._current_source = key
                self._handle_source_switch()

            # Hand the newest frame to the detection thread and move on. We
            # never wait for it: detection costs 50-650 ms against a 40 ms
            # frame budget, so blocking on it made playback ~0.1x real-time.
            # A 5 s clip taking a minute reads as a broken system regardless
            # of how good the labels are.
            with self._frame_lock:
                self._pending_frame = frame.image
                self._pending_ts = frame.timestamp

            annotations = self._last_annotations
            if self._on_frame:
                overlaid = draw_overlays(frame.image, annotations,
                                         (self._caption or {}).get("text"))
                self._on_frame(overlaid, annotations, self._caption)

            # --- throughput reporting ---
            fps_frames += 1
            now = time.time()
            self.stats["frames"] += 1
            self.stats["last_frame_at"] = now
            if now - fps_t0 >= 5.0:
                fps = fps_frames / (now - fps_t0)
                self.stats["playback_fps"] = round(fps, 1)
                log.info(f"playback {fps:.1f} FPS  ({len(self._last_ids)} tracked, "
                         f"detect={self._proc_ema*1000:.0f}ms)")
                fps_t0, fps_frames = now, 0
                # The source may have switched to one with another frame rate.
                target_dt = 1.0 / max(1, int(self._source.fps))

            # A file is paced to its frame rate here; a live camera paces
            # itself (read() waits for the next frame), and sleeping on top
            # would fall behind it and serve older and older frames.
            elapsed = time.time() - t0
            if elapsed < target_dt and not is_live(key or ""):
                time.sleep(target_dt - elapsed)

    def _process_tracks(self, image: np.ndarray, tracks: list[Track],
                        timestamp: float | None = None) -> list[dict]:
        active: set[int] = set()
        annotations: list[dict] = []
        h, w = image.shape[:2]

        for t in tracks:
            active.add(t.person_id)
            x1, y1, x2, y2 = map(int, t.xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue

            # What we hand the deep classifier depends on how many people are
            # in shot, because the two cases have opposite requirements.
            #
            # SINGLE PERSON -> the whole frame. Kinetics-400 models are
            # trained on complete scenes, and the scene IS the evidence: a
            # stove means cooking, a desk means typing. Measured directly
            # (scripts/exp_crop_vs_fullframe.py), full-frame raises in-vocab
            # coverage dramatically — writing 0.18 -> 0.86, cleaning
            # 0.21 -> 0.87, using-computer 0.64 -> 1.00 — and flipped
            # "writing" from a wrong answer to a correct one at 1.00.
            #
            # MULTIPLE PEOPLE -> a padded crop per person. A single frame
            # label cannot be attributed to one individual, so here we accept
            # weaker context in exchange for per-person answers.
            if len(tracks) == 1:
                crop = image
            else:
                pad_x = int((x2 - x1) * 0.25)
                pad_y = int((y2 - y1) * 0.25)
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(w, x2 + pad_x)
                cy2 = min(h, y2 + pad_y)
                crop = image[cy1:cy2, cx1:cx2]

            # Keypoints come from the tracker (YOLOv8-pose), already in
            # original-frame pixel coords. Crop is passed so RGB classifiers
            # (VideoMAE) can build their rolling clip.
            pred = self._classifier.update(t.person_id, t.keypoints, t.xyxy,
                                           crop, timestamp,
                                           single_person=len(tracks) == 1,
                                           frame_hw=(h, w))
            # A tracked person is never shown unlabelled. On the very first
            # frame of a track there is no motion estimate yet (a velocity
            # needs two samples), so we say so explicitly instead of
            # rendering a bare "-", which reads as a broken detector.
            label, conf = "detecting…", 0.0
            if pred is not None:
                smoothed = self._smoother.smooth(t.person_id, pred)
                if smoothed is not None:
                    smoothed = self._stabilizer.update(
                        t.person_id, smoothed,
                        timestamp if timestamp is not None else time.monotonic())
                    label, conf = smoothed.label, smoothed.confidence
                    self._events.update(t.person_id, smoothed, datetime.utcnow())

            annotations.append({
                "xyxy": t.xyxy,
                "person_id": t.person_id,
                "label": label,
                "confidence": conf,
                # Caption context (see VideoMAEClassifier.update).
                "detail": getattr(pred, "detail", None),
                "posture": getattr(pred, "posture", None),
            })

        for gone in self._last_ids - active:
            self._events.close_person(gone)
            self._smoother.reset(gone)
            self._stabilizer.reset(gone)
        self._last_ids = active

        return annotations
