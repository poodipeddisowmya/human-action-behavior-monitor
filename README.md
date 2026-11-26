# Human Action & Behavior Monitor

Finds every person in a video, follows them with a stable ID, recognises what
they are doing, names the people it has been taught to recognise, and logs
every activity with photo proof, live on a web dashboard. Reads CCTV / IP
cameras (RTSP), USB cameras or video files. Runs on any 64-bit Linux edge
board (an NVIDIA Jetson, a Raspberry Pi 5, a Rockchip board, a mini PC), a
laptop, a server or a cloud GPU: one installer sets it up for the machine.

- **Activity recognition:** 15 everyday activities, 10 postures and movements.
- **New activities without retraining:** *making tea* and *taking tablets*,
  defined in words; *falling down* raises an alert.
- **Train your own activities:** clips in a folder per activity, one
  command, minutes on the machine that runs the monitor, with a measured accuracy report
  ([guide](docs/TRAINING.md)); the example teaches *sleeping at desk*.
- **Face recognition:** "Asha is making tea"; anyone else stays Unknown, and
  can be named from the dashboard with one click (with consent).
- **Activity log with proof:** who, what, when, how long, which camera, and
  2–3 photos, written to the database the moment each activity ends; the
  **Database** tab shows the rows arriving live.
- **CCTV cameras:** one line per camera in `.env`; reconnects by itself;
  the password never leaves the server ([guide](docs/CCTV.md)).
- **Hardware comparison:** the same clips on a laptop CPU, a Tesla T4 GPU and
  a Jetson, side by side.

**Start here:** [Technical overview](docs/TECHNICAL_OVERVIEW.md), the whole
project explained simply · [Client demo](docs/CLIENT_DEMO.md), the demo
step by step · [CCTV cameras](docs/CCTV.md) · [Training new
activities](docs/TRAINING.md) · [Deployment](docs/DEPLOYMENT.md) on any
board, as a service · [Jetson](docs/JETSON.md).

---

## 1. Get the project

You need **Git** and **Git LFS**: the model files and videos (~550 MB) are
stored with Git LFS. Git for Windows includes it; on Linux `./setup.sh` installs it if
needed (or install `git-lfs` from your package manager).

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
| **Any Linux machine or edge board**: NVIDIA Jetson, Raspberry Pi 5, Rockchip RK3588, Intel / AMD mini PC, PC or server with or without an NVIDIA GPU | `./setup.sh` | `./run.sh` |
| Windows laptop or PC | double-click `setup.bat` | double-click `run.bat` |
| Windows with an NVIDIA GPU | `setup_gpu.bat` ([guide](docs/GPU_DEMO.md)) | `run.bat` |
| Docker (GPU server, or any CPU machine) | [guide](docs/DEPLOYMENT.md) | `docker compose up -d --build` |
| Kaggle (free cloud GPU) | [guide](docs/KAGGLE_DEMO.md) | one notebook cell |

`./setup.sh` looks at the machine and installs what fits it: CUDA and
TensorRT on a Jetson, CUDA on an NVIDIA GPU, and on anything else CPU-only
PyTorch plus the models compiled for that CPU (it measures which runtime is
fastest there). It needs no sudo, and fetches its own Python 3.11 if the
system's is unsuitable. Which boards, and what to expect on each:
[Deployment](docs/DEPLOYMENT.md). To start at every boot:
`scripts/install_service.sh`.

macOS: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`,
then `.venv/bin/python scripts/run_pipeline.py`.

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

It also fetches four clips of an Indian corporate office (desk work, computer
work, a phone call) for the Source menu's *Office* group. Without them
everything works, except the Overview's play buttons for those clips.

## 3. Using it

| To | Do |
|---|---|
| Watch a clip | **Source** menu (top right): clips grouped as New activities, Activities, Scenes |
| Analyse your own video | **Upload** (mp4, avi, mov, mkv, webm; up to 500 MB) |
| Use a CCTV camera | Add `CAMERA_RECEPTION=rtsp://…` to `.env`, then pick *Reception* in **Source**, or start with `--source Reception` ([guide](docs/CCTV.md)) |
| Use another camera | **Live camera** (USB camera on this computer), **Go live** (your browser's webcam), or start with `--source 0` |
| Watch the database fill | **Database** tab |
| Record what the dashboard shows | `python scripts/record_live.py --seconds 30` |
| Teach it a new activity | Clips in `data/training/<activity>/`, then `python scripts/train_activities.py` ([guide](docs/TRAINING.md)) |
| Name a person | **Faces on screen**: type a name under their face; or **Known people → Add person** with 2–5 photos |
| See the proof of an activity | Activity log → click a photo |
| Compare the hardware | **CPU · T4 · Jetson** tab |

## 4. Results

Measured on the same 29 labelled clips, 12 s each, as the screen showed them:

| | Laptop CPU (earlier version) | ARM board CPU, no GPU | Tesla T4 | Jetson Orin Nano Super |
|---|---:|---:|---:|---:|
| Activities clearly recognised | 1/29 | 15/29 | 21/29 | 22/29 |
| Correct label on screen | 4% | 47% | 66% | 68% |
| Playback speed | 18 fps | 22 fps | 24 fps | 22 fps |

A GPU is what makes it most accurate: the activity model looks several
times a second. Without one, the models compiled for the CPU by setup make
a board usable (the ARM column: a 6-core Cortex-A78AE, the Jetson's CPU with
its GPU unused; the laptop was recorded before compilation existed).
Which machine to choose: [Deployment](docs/DEPLOYMENT.md). Accuracy, limits
and how to improve it on your own cameras:
[Technical overview §10](docs/TECHNICAL_OVERVIEW.md#10-accuracy-and-limits).

## 5. Configuration

- **`.env`** (copy from `.env.example`): `ACCESS_TOKEN` to password-protect
  a server others can reach; `API_PORT`; `DEVICE` (auto, cuda, cpu);
  `FACE_ID`, `OPEN_VOCAB`, `CUSTOM_ACTIVITIES`, `EVIDENCE` to switch features;
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
| Slow or stuttering video | On a CPU, run `python scripts/optimise_for_device.py` once (setup.sh does). Still slow: a GPU or Jetson fixes it. |
| Names never appear | Nobody enrolled, or faces too small or turned away. |
| Tea / tablets never appear | They need a GPU; check the log for `[open-vocab]`. |

Logs: `logs/monitor.log`.

## 7. Project layout

```
├── README.md, setup.sh, run.sh,          start here; install and start (Linux, any board)
│   setup*.bat, run.bat, setup_jetson.sh  Windows; Jetson (setup.sh calls it)
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
