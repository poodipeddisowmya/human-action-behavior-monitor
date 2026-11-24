# Architecture

For a plain-language explanation of the whole project, start with
[TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md). This page is the developer's
view: the stages, the modules, the threads, and where to extend it.

## End-to-end flow

```
 video source ──► YOLOv8-pose ──► ByteTrack ──► per person, per frame:
 (file, camera,   (people +       (stable IDs)
  upload, the     17 keypoints)                 ┌─ skeleton rules ─────────── posture, motion
  viewer's                                      ├─ activity model ─────────── 15 activities
  webcam)                                       │   (VideoMAE on GPU, S3D on CPU)
                                                ├─ object context ─────────── cup, phone, laptop… in hand
                                                ├─ CLIP ───────────────────── place; sitting/standing check
                                                ├─ X-CLIP (GPU) ───────────── activities from a text description
                                                ├─ fall detector ──────────── upright → on the ground
                                                └─ face recognition ───────── who (enrolled people)
                                                         │
                                consensus voting ─► smoothing ─► display stabiliser
                                                         │
                     ┌───────────────────────────────────┼────────────────────────────┐
                     ▼                                   ▼                            ▼
              event generator                      subtitles                annotated video
         (one row per activity,              (template sentence;          (boxes, names, labels)
          with 2–3 proof photos)              SmolVLM2 on a T4)
                     │                                   │                            │
                     ▼                                   ▼                            ▼
            SQLite + data/evidence/              WebSocket push                 MJPEG stream
                     └──────────────────────► React dashboard ◄─────────────────────┘
```

## Modules

| Folder | What it does |
|---|---|
| `src/input/` | Video sources behind one interface: files, CCTV / IP cameras (RTSP, HTTP; reconnecting), USB cameras, the viewer's browser webcam; switchable at runtime without reloading models. |
| `src/tracking/` | YOLOv8-pose detection (PyTorch, or a TensorRT engine on a Jetson) and ByteTrack IDs. |
| `src/action_recognition/` | Skeleton rules, the Kinetics activity model restricted to the target vocabulary, object context, CLIP posture check, open-vocabulary X-CLIP, fall detector. |
| `src/identity/` | Face recognition (YuNet + SFace) and enrolment, with its safeguards. |
| `src/temporal/` | Smoothing, display stabilisation, and turning labels into events. Pure logic, unit-tested without models. |
| `src/description/` | Subtitles: place recognition (CLIP), template captions, the SmolVLM2 narrator on a T4-class GPU. |
| `src/storage/` | Async SQLAlchemy models (people, activities, enrolled faces) and proof images. |
| `src/api/` | FastAPI: REST, WebSocket, MJPEG video, optional access token; serves the built dashboard. |
| `src/pipeline/` | The orchestrator: runs every stage per frame and owns their state. |
| `src/utils/` | Drawing, time formatting, the recorded comparison summary. |
| `ui/` | React + Vite + Tailwind dashboard. `ui/dist/` is committed, so running it needs no Node.js. |
| `config/` | `model_config.yaml` (models and calibrated thresholds, with the measurements behind them) and `settings.py` (environment variables, hardware profiles). |

## Threads

- **Pipeline thread:** reads frames, detects and tracks, runs the per-person
  stages, draws the frame.
- **Detection runs decoupled from display**, so a slow detection never
  stalls the video; labels attach to the newest frame.
- **Background workers**, one each, never blocking the video: the activity
  model's looks, X-CLIP, CLIP place recognition, face recognition, the
  SmolVLM2 narrator.
- **FastAPI on the asyncio loop.** `src/api/runtime.py` bridges the two:
  events are written to the database with `run_coroutine_threadsafe`,
  frames are handed over through a lock-protected buffer.

## Hardware profiles

`config/settings.py` picks one at start-up; each is a block in
`config/model_config.yaml`.

| Profile | When | Detector | Activity model | Extras |
|---|---|---|---|---|
| CPU | no usable GPU | YOLOv8n-pose, 416 px | S3D, a look every 1.5 s | template subtitles |
| GPU | CUDA GPU (Tesla T4, RTX) | YOLOv8s-pose, 640 px, FP16 | VideoMAE, a look every 0.4 s | X-CLIP, SmolVLM2 subtitles |
| Jetson | `/etc/nv_tegra_release` exists | TensorRT engine, FP16 | VideoMAE FP16 | X-CLIP; template subtitles (memory) |

## Data

| What | Where | Kept |
|---|---|---|
| Activity log | `logs/monitor.db` (SQLite; `DATABASE_URL` for PostgreSQL) | until deleted |
| Proof photos | `data/evidence/<date>/` | `EVIDENCE_RETENTION_DAYS` (30) |
| Enrolled faces | `data/faces/` and the database | until the person is deleted |
| Faces of people not enrolled | memory only, while in view | never written |

`data/evidence/`, `data/faces/` and the database are git-ignored.

## Extending it

| To add | Do this |
|---|---|
| An activity Kinetics-400 knows | One line in `src/action_recognition/target_vocabulary.py` |
| An activity described in words | One entry under `open_vocab.activities` in `config/model_config.yaml` ([NEW_ACTIVITIES.md](NEW_ACTIVITIES.md)) |
| A person to recognise | The dashboard: Faces on screen, Known people, or a proof photo |
| A place for the subtitles | One line in `_PLACES` in `src/description/scene.py` |
| A CCTV / IP camera | One `CAMERA_<NAME>=rtsp://…` line in `.env` ([CCTV.md](CCTV.md)) |
| Another kind of video input | A capture class beside `StreamCapture` (`src/input/network_camera.py`), opened in `SwappableFileSource._open` |
| Another database | `DATABASE_URL` (async SQLAlchemy 2) |
| Another hardware device in the comparison | `scripts/add_showcase_side.py` ([JETSON.md](JETSON.md)) |
