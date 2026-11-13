# Architecture

## End-to-end flow

```
VideoSource ─▶ PersonTracker(YOLOv8+ByteTrack) ─▶ per-track crop ─▶ PoseEstimator(MediaPipe)
                                                                          │
                                              ┌───────────────────────────┘
                                              ▼
                                  ActionClassifier (ST-GCN / heuristic)
                                              │
                                              ▼
                                      TemporalSmoother
                                              │
                                              ▼
                                       EventGenerator ──▶ Runtime
                                                             │
                                       ┌─────────────────────┼─────────────────────┐
                                       ▼                     ▼                     ▼
                                  SQLite log            WebSocket push        MJPEG stream
                                                             │                     │
                                                             ▼                     ▼
                                                        UI event feed        UI video panel
```

## Threading & async model

- Pipeline runs in a **background thread** (heavy CV work, blocking IO).
- FastAPI runs on the **asyncio loop**.
- `Runtime` bridges the two: thread-safe frame buffer + `run_coroutine_threadsafe` for DB writes.

## Why modules are separated

| Module | Reason it's a boundary |
|---|---|
| `input/` | Swap dataset ↔ file ↔ camera without touching the pipeline. |
| `detection/` vs `tracking/` | Detector alone useful for evaluation; tracker adds ID persistence. |
| `action_recognition/` | Pose and classifier are independent — either can be replaced. |
| `temporal/` | Smoother + event generator are pure logic; unit-testable without models. |
| `storage/` | DB engine can move from SQLite → Postgres by changing one URL. |
| `api/` | Runtime is decoupled from HTTP; can be reused by a CLI or gRPC. |
| `ui/` | Fully separate app; talks to backend only via REST + WebSocket. |

## Extensibility hooks

- **New action**: add to `config/model_config.yaml`, retrain / update classifier.
- **New model**: implement the same 2–3-method interface, register in orchestrator.
- **New input**: subclass `VideoSource`.
- **Different DB**: change `DATABASE_URL`; models are async-SQLAlchemy 2.x.

## Resource budget (default profile)

| Stage | Cost per frame (RTX 3060 / 720p) |
|---|---|
| YOLOv8n + ByteTrack | ~6 ms |
| MediaPipe Pose (per person) | ~5 ms |
| ST-GCN window (every 8 frames) | ~2 ms |
| Draw + JPEG encode | ~2 ms |
| **Total** | ~15 ms → **60+ FPS single-person, 25+ FPS three-person** |

CPU-only fallback runs at ~10–12 FPS on a modern laptop.
