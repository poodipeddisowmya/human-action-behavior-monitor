# Human Action & Behavior Monitor

Finds every person in a video, follows them with a stable ID, recognises what
they are doing, names the people it has been taught to recognise, and logs
every activity with photo proof, live on a web dashboard. Runs on a laptop, a
cloud GPU, or an NVIDIA Jetson on site.

- **Activity recognition:** 15 everyday activities, 10 postures and movements.
- **New activities without retraining:** *making tea* and *taking tablets*,
  defined in words; *falling down* raises an alert.
- **Face recognition:** "Asha is making tea"; anyone else stays Unknown, and
  can be named from the dashboard with one click (with consent).
- **Activity log with proof:** who, what, when, how long, and 2–3 photos.
- **Hardware comparison:** the same clips on a laptop CPU, a Tesla T4 GPU and
  a Jetson, side by side.

**Start here:** [Technical overview](docs/TECHNICAL_OVERVIEW.md), the whole
project explained simply · [Client demo](docs/CLIENT_DEMO.md), the
10-minute demo step by step.

---

## 1. Get the project

You need **Git** and **Git LFS**: the model files and videos (~550 MB) are
stored with Git LFS. Git for Windows includes it; on Linux install
`git-lfs` from your package manager; on a Jetson `setup_jetson.sh` installs it.

```bash
git lfs install
git clone https://github.com/poodipeddisowmya/human-action-behavior-monitor.git
cd human-action-behavior-monitor
```

Do not use GitHub's "Download ZIP": it leaves small pointer files instead of
the models and videos. If you did, run `git lfs pull` in a clone.

## 2. Install and start

Python **3.10, 3.11 or 3.12** ([3.11 recommended](https://www.python.org/downloads/release/python-3119/);
3.13 and newer cannot install one of the pinned libraries).

| Machine | Install once | Start |
|---|---|---|
| Windows laptop or PC | double-click `setup.bat` | double-click `run.bat` |
| Windows with an NVIDIA GPU | `setup_gpu.bat` ([guide](docs/GPU_DEMO.md)) | `run.bat` |
| Linux or macOS | see below | `python scripts/run_pipeline.py` |
| NVIDIA Jetson | `./setup_jetson.sh` ([guide](docs/JETSON.md)) | `./run_jetson.sh` |
| Kaggle (free cloud GPU) | [guide](docs/KAGGLE_DEMO.md) | one notebook cell |
| GPU server (Docker) | [guide](docs/DEPLOYMENT.md) | `docker compose up -d --build` |

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_install.py
python scripts/run_pipeline.py
```

Then open **http://localhost:8000** (from another computer:
`http://<this-computer's-address>:8000`). The dashboard opens on the
**Overview**. You do not need Node.js: the dashboard is pre-built.

`scripts/verify_install.py` checks everything (Python, packages, the large
files, the model) and prints the fix for any problem.

### The new-activity demo clips

The clips for making tea, taking tablets and falling down are downloaded
from their original sources, because their licences do not allow re-sharing
them:

```bash
python scripts/fetch_new_activity_clips.py
```

Without them everything works, except the Overview's play buttons for those
three activities.

## 3. Using it

| To | Do |
|---|---|
| Watch a clip | **Source** menu (top right): clips grouped as New activities, Activities, Scenes |
| Analyse your own video | **Upload** (mp4, avi, mov, mkv, webm; up to 500 MB) |
| Use a camera | **Live camera** (USB camera on this computer), **Go live** (your browser's webcam), or start with `--source 0` |
| Name a person | **Faces on screen**: type a name under their face; or **Known people → Add person** with 2–5 photos |
| See the proof of an activity | Activity log → click a photo |
| Compare the hardware | **CPU · T4 · Jetson** tab |

## 4. Results

Measured on the same 29 labelled clips, 12 s each, as the screen showed them:

| | Laptop CPU | Tesla T4 | Jetson Orin Nano Super |
|---|---:|---:|---:|
| Activities clearly recognised | 1/29 | 21/29 | 22/29 |
| Correct label on screen | 4% | 66% | 68% |
| Playback speed | 18 fps | 24 fps | 22 fps |

A GPU is what makes it accurate: the activity model looks several times a
second instead of once every 1.5 s. Accuracy, limits and how to improve it
on your own cameras: [Technical overview §10](docs/TECHNICAL_OVERVIEW.md#10-accuracy-and-limits).

## 5. Configuration

- **`.env`** (copy from `.env.example`): `ACCESS_TOKEN` to password-protect
  a server others can reach; `API_PORT`; `DEVICE` (auto, cuda, cpu);
  `FACE_ID`, `OPEN_VOCAB`, `EVIDENCE` to switch features;
  `EVIDENCE_RETENTION_DAYS` (default 30).
- **`config/model_config.yaml`:** models and thresholds. Each value's
  comment says what it was measured against; read it before changing one.
  New activities described in words go under `open_vocab`.

## 6. If something goes wrong

Run `python scripts/verify_install.py` first (`.venv\Scripts\python.exe` on
Windows). Then:

| Symptom | Fix |
|---|---|
| `unsupported Python version` | Install Python 3.11, delete `.venv`, run setup again. |
| "large files … are LFS pointers" | Install Git LFS, then `git lfs install` and `git lfs pull`. |
| Page does not load | Models take ~10–30 s to load; wait, then refresh. |
| Port 8000 in use | Set `API_PORT=8001` in `.env`. |
| Slow or stuttering video | Expected on an older CPU; a GPU or Jetson fixes it. |
| Names never appear | Nobody enrolled, or faces too small or turned away. |
| Tea / tablets never appear | They need a GPU; check the log for `[open-vocab]`. |

Logs: `logs/monitor.log`.

## 7. Project layout

```
├── README.md, setup*.bat, run.bat,       start here; install and start
│   setup_jetson.sh, run_jetson.sh
├── config/            settings.py (environment), model_config.yaml (models, thresholds)
├── docs/              technical overview, demo, architecture, models, guides
├── src/
│   ├── input/         video sources: files, cameras, browser webcam
│   ├── tracking/      person detection (YOLOv8-pose) and IDs (ByteTrack)
│   ├── action_recognition/   activities, postures, objects, falls, text-defined activities
│   ├── identity/      face recognition and enrolment
│   ├── temporal/      smoothing, display stability, activity events
│   ├── description/   subtitles and place recognition
│   ├── storage/       database and proof photos
│   ├── api/           web server, API, video stream
│   └── pipeline/      runs every stage per frame
├── ui/                dashboard (React); ui/dist is pre-built
├── models/            model weights (Git LFS)
├── data/samples/      demo clips (Git LFS); uploads land here
├── data/showcase/     the recorded CPU / T4 / Jetson comparison (Git LFS)
├── scripts/           tools: install check, evaluation, recording ([list](scripts/README.md))
├── kaggle/            cloud GPU notebooks
├── tests/             automated tests
└── Dockerfile, docker-compose.yml
```

## 8. Development

```bash
python -m pytest tests -q                    # automated tests
python scripts/eval_offline.py --realtime    # accuracy on the demo clips
cd ui && npm install && npm run build        # after changing the dashboard
```

Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · models and
licences: [docs/MODEL_SELECTION.md](docs/MODEL_SELECTION.md) · API:
[docs/API.md](docs/API.md), or `/docs` on a running server.

## Licences

Third-party models and clips keep their own licences
([models](docs/MODEL_SELECTION.md#licences-to-check-before-a-commercial-deployment),
[clips](data/showcase/CREDITS.md)). Before a commercial product: YOLOv8 is
AGPL-3.0, and the VideoMAE weights used on GPUs are non-commercial.
