# Human Action & Behavior Monitor

Detects people in video, tracks them with stable IDs, recognises what they are
doing, and streams the results to a live monitoring dashboard.

Works on bundled demo clips, on any video you upload through the UI, or on a
webcam — the server's own, or the viewer's own browser webcam.

**GPU:**
[Client demo on Kaggle's free GPU](docs/KAGGLE_DEMO.md) ·
[Production deployment (Docker)](docs/DEPLOYMENT.md) ·
[Windows GPU machine](docs/GPU_DEMO.md) ·
[NVIDIA Jetson](docs/JETSON.md) ·
[Client demo on the Jetson](docs/CLIENT_DEMO.md)

---

## Setup (Windows) — do this once

You need **Python 3.10, 3.11 or 3.12** — [3.11 is the tested
version](https://www.python.org/downloads/release/python-3119/). Tick
**"Add Python to PATH"** during installation.

> **Not 3.13 or newer.** One of the pinned libraries has no prebuilt package
> for 3.13+, so the install tries to compile from source and fails. `setup.bat`
> checks this first and tells you, rather than letting you find out ten minutes
> in.

Then double-click:

```
setup.bat
```

It creates a virtual environment, installs everything, and runs a check that
tells you plainly if anything is missing. Takes a few minutes the first time,
mostly downloading PyTorch.

**You do not need Node.js or npm.** The dashboard is pre-built and served by
the same Python process.

### Start it

Double-click:

```
run.bat
```

A browser opens at **http://localhost:8000**. That is the whole application —
video, dashboard, and API on one port.

Press `Ctrl+C` in the black window to stop it.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_install.py
python scripts/run_pipeline.py
```

---

## Using it

- **Switch clips** — the *Source* dropdown, top right. Switching is instant;
  the models stay loaded.
- **Upload your own video** — the *Upload* button. `.mp4`, `.avi`, `.mov`,
  `.mkv`, `.webm`, up to 500 MB. It starts playing immediately.
- **Webcam** — `python scripts/run_pipeline.py --source 0`
- **A specific file** — `python scripts/run_pipeline.py --source path\to\video.mp4`

Each tracked person gets a coloured box, a stable ID, the recognised action and
a confidence figure. Completed actions accumulate in the events table with
their start time and duration.

The dashboard opens on an **Overview** of the four things it does (the
hardware comparison, the new activities, face recognition, the activity log
with proof photos), each with a button that shows it working; the live view
is at `#live`.

The **CPU · T4 · Jetson** tab (`http://localhost:8000/#compare`) plays a recorded
comparison: the same clips through the same software on a laptop CPU, a Tesla
T4 GPU and an NVIDIA Jetson, side by side and frame-synced, with the measured
figures. Install a recording with `python scripts/install_showcase.py
dist/client-demo`; add another device with `scripts/add_showcase_side.py`
(see [docs/JETSON.md](docs/JETSON.md)).

On a GPU the subtitle is a detailed description written by a small
vision-language model (SmolVLM2) from the frames themselves, grounded in what
the system detected; on a CPU it is the template sentence built from the
labels ("A person is cooking in a kitchen.").

---

## What it recognises

Fifteen target activities, from a model pretrained on Kinetics-400:

`brushing teeth` · `cleaning` · `cooking` · `cycling` · `dancing` · `drinking` ·
`driving` · `eating` · `exercising` · `reading` · `smoking` · `swimming` ·
`using computer` · `using phone` · `writing`

Each is a group of Kinetics classes ([target_vocabulary.py](src/action_recognition/target_vocabulary.py)):
"dancing" covers fifteen dance styles, "swimming" the three strokes. Adding an
activity Kinetics already knows is one line there; one it does not know needs
training on labelled footage.

On a GPU, three more, which the Kinetics model does not know
([docs/NEW_ACTIVITIES.md](docs/NEW_ACTIVITIES.md)):

`making tea` · `taking tablets` — recognised from a text description by a
zero-shot video transformer (X-CLIP); adding another is one line of config.
`falling down` — from the skeleton over time (upright, then lying within a
second), shown as an alert in the subtitles and the events log.

Every logged activity keeps 2–3 proof images of the person (start, middle,
end), shown in the events table.

**Face recognition** names enrolled people ("Ramesh is taking tablets"):
enrol them with a few photos in the Known people panel, or straight from a
proof image; everyone else stays Unknown. With consent only — see
[docs/FACE_RECOGNITION.md](docs/FACE_RECOGNITION.md).

Plus body posture and motion from skeleton geometry, which works even when no
target activity is in view:

`standing` · `sitting` · `walking` · `running` · `lying` · `kneeling` ·
`bending` · `crawling` · `push-up` · `falling`

### Honest accuracy note

Measured 2026-11-04 on the 15 labelled demo clips, **as the dashboard shows
them** (real-time playback, `scripts/live_accuracy.py`):

| Hardware | Correct | Correct label on screen after |
|---|---|---|
| NVIDIA Tesla T4 (Kaggle) | **15/15** | 0.5–2.3 s (cycling 5.8 s) |
| Laptop CPU | 11/15 | 4–10 s; misses eating, smoking, cycling, walking |

A GPU matters for accuracy, not just speed: the activity model gets several
looks a second instead of one every 1.5 s, so its votes agree sooner.

The offline harness below scores the first 120 frames per clip; on a CPU it
runs slower than real time, which flatters the CPU (it gave 14/15 there).
Use `--realtime` for numbers comparable to the live view:

```bash
python scripts/eval_offline.py --realtime                  # offline harness
python scripts/live_accuracy.py --base http://127.0.0.1:8000  # against a running server
```

These are demo clips the system was tuned on. On footage it has never seen
(the four held-out phone clips) it scores 3/4 — it reads a drinking clip as
sitting. The largest remaining improvement is **fine-tuning on real footage
from the deployment site**; run that on a GPU once a few hundred labelled
clips exist.

Playback runs at roughly real time on a normal laptop CPU (measured 15–22 FPS
on 25 fps clips) and on a T4 GPU (detection 14–17 ms per frame).

---

## If something goes wrong

Run the diagnostic first — it checks every dependency, weight file and folder
permission, and prints a specific fix for each problem:

```bash
.venv\Scripts\python.exe scripts\verify_install.py
```

| Symptom | Cause and fix |
|---|---|
| `Python not found` | Python missing from PATH. Reinstall with "Add Python to PATH" ticked. |
| `unsupported Python version` | You have 3.13+. Install 3.11, delete the `.venv` folder, run `setup.bat` again. |
| Install fails compiling numpy | Same cause as above — wrong Python version. |
| Page will not load | Server still starting — model loading takes ~10 s. Wait, then refresh. |
| Port 8000 already in use | Something else has the port. `set API_PORT=8001` then `run.bat`. |
| Boxes but no action labels | The detector must be a `-pose` model. Check `detector_weights` in `config/model_config.yaml`; the app prints a loud warning if it is wrong. |
| Slow / stuttering video | Expected on an older CPU. Lower `detector_imgsz` to `320` in `config/model_config.yaml`. |

Logs are written to `logs/monitor.log`.

---

## Configuration

`config/model_config.yaml` — models and thresholds. The comments explain what
each value was measured against; several are deliberately calibrated numbers
rather than round guesses, so read the note before changing one.

`.env` (optional, copy from `.env.example`) — `API_PORT`, `DEVICE`,
`TARGET_FPS`, and `ACCESS_TOKEN` to password-protect a server reachable from
other machines. Full list: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## How it works

```
video ─► YOLOv8-pose ─► ByteTrack ─► ┬─► skeleton rules ──┐
        (detect +        (stable     │   (posture/motion) ├─► consensus ─► smoothing ─► events ─► UI
         17 keypoints)    IDs)       └─► S3D / Kinetics ──┘     voting
                                         (11 activities)
```

Two classifiers run against every tracked person. The skeleton rules are cheap
and read geometry directly. The deep model reads pixels and scene context, and
its 400 Kinetics classes are masked down to the 11 target activities — which
yields both a confidence *and* a coverage figure saying whether the scene is
one the system claims to handle at all. Results are pooled by weighted
consensus over a rolling window, so a single noisy frame cannot flip a label.

Full detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
[docs/MODEL_SELECTION.md](docs/MODEL_SELECTION.md) ·
[docs/API.md](docs/API.md)

---

## Project layout

```
├── setup.bat / run.bat      # one-time setup, and start
├── config/                  # settings + model_config.yaml
├── models/                  # bundled weights (YOLOv8-pose, S3D) — no download needed
├── data/samples/            # demo clips, and where uploads land
├── src/
│   ├── input/               # video source abstraction (file / camera)
│   ├── tracking/            # YOLOv8-pose + ByteTrack
│   ├── action_recognition/  # skeleton rules + S3D, restricted vocabulary
│   ├── temporal/            # smoothing + event generation
│   ├── storage/             # SQLite via SQLAlchemy
│   ├── api/                 # FastAPI, WebSocket, MJPEG, serves the UI
│   └── pipeline/            # orchestrator
├── ui/                      # React dashboard (dist/ is pre-built)
├── scripts/                 # run, verify, evaluate, benchmark, batch-process, Kaggle launcher
├── kaggle/                  # kaggle_demo.ipynb — one-click GPU demo
├── Dockerfile, docker-compose.yml   # GPU production deployment
└── tests/                   # pytest
```

## Development

```bash
.venv\Scripts\python.exe -m pytest tests -q      # tests
.venv\Scripts\python.exe scripts\eval_offline.py # accuracy on the demo set
```

Changing the dashboard needs Node:

```bash
cd ui && npm install && npm run build
```

The Python server picks up `ui/dist` automatically on restart.
