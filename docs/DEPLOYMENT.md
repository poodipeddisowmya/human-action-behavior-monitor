# Deploying it: any edge board, PC or server

The same code runs on every machine below. `./setup.sh` looks at the
machine and installs what fits it; the monitor then picks its settings from
what it finds (GPU or CPU, Jetson or not) with no file to edit.

## 1. Which machine

| Machine | How it runs | Tested | Recognised (29 demo clips) |
|---|---|---|---|
| **NVIDIA Jetson Orin** (Nano 8 GB, NX, AGX; JetPack 6) | GPU: CUDA, TensorRT, VideoMAE; every feature | Orin Nano Super ([JETSON.md](JETSON.md)) | 22/29 |
| **PC or server with an NVIDIA GPU** (4 GB+; Linux or Windows) | GPU: CUDA, VideoMAE; every feature | Tesla T4 | 21-23/29 |
| **64-bit ARM board, no NVIDIA GPU** (Raspberry Pi 5 8 GB, Rockchip RK3588 boards such as Orange Pi 5 or Rock 5, NXP i.MX 8M Plus...) | CPU: models compiled for the board ([below](#2-on-a-cpu-the-models-are-compiled-for-the-machine)) | a 6-core Cortex-A78AE CPU, the Jetson's own with its GPU unused, standing in for an ARM board | 15/29 |
| **Intel / AMD mini PC, industrial PC, laptop** (x86_64, no NVIDIA GPU) | CPU: models compiled for the machine | a laptop, before compilation existed | 1/29 then; not yet measured with compilation |

**Not supported:** 32-bit operating systems (PyTorch needs 64-bit: on a
Raspberry Pi, install the 64-bit Raspberry Pi OS); less than 4 GB of RAM
(the CPU profile peaks at about 2.2 GB); Jetson Nano, TX2 and Xavier
(JetPack 4 / 5: Python and PyTorch too old); the Orin Nano 4 GB (not enough
memory for the GPU models).

**What needs an NVIDIA GPU:** activities described in words (making tea,
taking tablets), detailed subtitles, and the more accurate VideoMAE
activity model. Everything else runs on a CPU: the activities and postures,
falling down, faces and names, activities trained on your own clips,
places in the subtitle, the activity log, proof photos, the database,
CCTV cameras.

**Accelerator chips** on some boards (the Rockchip NPU, Hailo, Coral, an
Intel iGPU) are not used: each needs its own model conversion. Those boards
run on their CPU, as above.

## 2. On a CPU, the models are compiled for the machine

PyTorch is the slowest way to run the two models that work all the time.
`setup.sh` runs `scripts/optimise_for_device.py`, which compiles the person
detector and the activity model with each runtime available (OpenVINO, ONNX
Runtime), times each against PyTorch on frames from the demo clips, checks
that it gives the same answers, and keeps the fastest. Which one wins depends
on the CPU, so it is measured, not assumed. On the ARM CPU above, 4 threads:

| | PyTorch | OpenVINO | ONNX Runtime | Kept |
|---|---:|---:|---:|---|
| Person detector, 416 px, per frame | 200 ms | **56 ms** | - | OpenVINO |
| Activity model (S3D), per look | 1,850 ms | 30,627 ms | **524 ms** | ONNX Runtime |

Same answers: the compiled activity model's scores differ from PyTorch's by
under 0.00001, and the compiled detector finds the same people in 96% of
cases (the rest: a box at the edge of the picture, from the square input).

On the 29 demo clips (`scripts/record_showcase.py`, the set used for every
machine): **15/29 clearly recognised, 47% correct label on screen, 22 fps**,
recorded from a fresh `./setup.sh` install. Laptop CPU before compilation:
1/29, 4%; Jetson: 22/29, 68%. Live on the same ARM CPU, 15 labelled demo
clips scored as the screen showed them (`scripts/live_accuracy.py`),
PyTorch against compiled:

| | PyTorch | Compiled |
|---|---:|---:|
| Clips recognised | 4/15 (27%) | **12/15 (80%)** |
| Person detection per frame, with the activity model running | 584 ms | 140 ms |
| Memory (peak) | 1.9 GB | 2.2 GB |

The compiled files go to `models/compiled/` (made per machine, not in
git). Re-run `python scripts/optimise_for_device.py` after changing
`detector_weights` or `detector_imgsz`; `--remove` goes back to PyTorch, as
does `CPU_RUNTIME=pytorch` in `.env`. A compiled model that fails to load is
skipped with a message and the PyTorch one runs: it can never stop the
monitor. On a slow CPU the activity model also looks less often, by itself,
so that the video keeps moving.

## 3. Install on the machine (recommended for a board)

```bash
git clone https://github.com/poodipeddisowmya/human-action-behavior-monitor.git
cd human-action-behavior-monitor
./setup.sh                      # once: 10-20 minutes, no sudo
./run.sh                        # dashboard on http://<board-address>:8000
scripts/install_service.sh      # start at every boot, restart if it stops
```

`setup.sh`, step by step:

1. reads the machine (board, CPU, RAM, disk, NVIDIA GPU); on a Jetson it
   hands over to `setup_jetson.sh` (CUDA and TensorRT from JetPack)
2. Git LFS for the models and demo clips (installs `git-lfs` into
   `~/.local/bin` if missing, checksum-checked)
3. a Python 3.10-3.12 environment in `.venv`; if the system has none (Debian
   13 and Raspberry Pi OS "Trixie" ship 3.13, some images lack `venv`), it
   fetches Python 3.11 with [uv](https://docs.astral.sh/uv/) into your home
   folder
4. packages: PyTorch with CUDA for an NVIDIA GPU; otherwise CPU-only PyTorch
   (PyPI's default would add ~3 GB of CUDA libraries the board cannot use)
   plus OpenVINO and ONNX Runtime
5. model weights this machine uses (`scripts/download_models.py`)
6. compiles the models for this machine (section 2)
7. creates `.env` from `.env.example`, and runs `scripts/verify_install.py`

Re-running it is safe. Afterwards the machine needs no internet. The one
system package it may ask for, on minimal images, is OpenCV's:
`sudo apt install -y libgl1 libglib2.0-0` (it tells you).

As a service, `systemctl --user status action-monitor` shows it,
`journalctl --user -u action-monitor -f` its output; run
`sudo loginctl enable-linger $USER` once so it starts at boot without
anyone logging in.

## 4. Docker

```bash
echo "ACCESS_TOKEN=$(openssl rand -hex 16)" > .env
docker compose up -d --build                                  # NVIDIA GPU server
docker compose --profile cpu up -d --build monitor-cpu        # CPU machine or board, x86_64 or arm64
docker compose logs -f
```

The GPU image needs the NVIDIA driver and the [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html);
any NVIDIA GPU with 4 GB or more (RTX 3050 upwards, T4, L4, A10). The CPU
image compiles the models for the machine on its first start (a few
minutes; kept on a volume). On a Jetson, install natively (section 3).
The images have not been built on this project's test board; section 3 is
the tested path on a board.

Open `http://<server>:8000/?token=<ACCESS_TOKEN>`. Put it behind HTTPS (a
reverse proxy such as Caddy or nginx, or a Cloudflare tunnel) if viewers
will use **Go live** with their own webcam: browsers allow cameras only on
https or localhost. The event database, logs and uploaded clips are on
named volumes and survive rebuilds. Docker restarts the container if it
exits, and the image's `HEALTHCHECK` marks it unhealthy if the server stops
answering.

## 5. Windows

`setup.bat` then `run.bat`; with an NVIDIA GPU `setup_gpu.bat`
([GPU_DEMO.md](GPU_DEMO.md)). To start at logon:
`scripts\install_autostart.ps1`. Add `ACCESS_TOKEN=...` to `.env` if the
machine is reachable from a network you don't control.

## Configuration

All settings are environment variables (or `.env`); models and thresholds
are in `config/model_config.yaml`.

| Variable | Default | Purpose |
|---|---|---|
| `ACCESS_TOKEN` | empty (open) | Required to view the dashboard or call the API. Set it for anything not on localhost. |
| `DEVICE` | `auto` | `auto` uses a working GPU, else CPU. A GPU that fails a real CUDA op falls back to CPU with a warning. |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | Bind address. |
| `TARGET_FPS` | `15` | |
| `STREAM_FPS` | `20` | Max frames/s sent to each browser. |
| `STREAM_MAX_WIDTH` | `1280` | Video is downscaled to this width for viewers (bandwidth). |
| `STREAM_JPEG_QUALITY` | `80` | |
| `DETECTOR_WEIGHTS` | from YAML | Override the person/pose detector, e.g. `yolov8s-pose.pt`. Downloaded into `models/` once; falls back to the bundled model if offline. |
| `DETECTOR_IMGSZ` | from YAML | Detector input size. |
| `PROBE_CAMERAS` | on for Windows / desktops | Look for USB cameras attached to the server. |
| `DATABASE_URL` | SQLite in `logs/` | Any async SQLAlchemy URL (e.g. Postgres). |
| `SOURCE` | first demo clip | What the monitor plays when started without `--source` (the service): a CCTV camera's name, a camera index, a file. |
| `CPU_RUNTIME` | `auto` | `pytorch` ignores the models compiled for this CPU. |
| `FACE_ID`, `OPEN_VOCAB`, `CUSTOM_ACTIVITIES`, `EVIDENCE` | `1` | Switch a feature off with `0`. |

On a GPU, the `gpu:` block of `model_config.yaml` applies automatically:
full-resolution detection, FP16 detector, VideoMAE, activity model 2.5×
a second; on a Jetson the `jetson:` block on top. On a CPU, the `model:`
values: 416 px detection, S3D, an activity look at most every 1.5 s.

## Operations endpoints

| Endpoint | Auth | Meaning |
|---|---|---|
| `GET /api/health` | none | Liveness: process answers. For probes. |
| `GET /api/ready` | token | 200 only when models are loaded **and** frames are flowing; 503 with a reason otherwise. |
| `GET /api/metrics` | token | FPS, detection latency, people tracked, viewers, restarts, errors, GPU memory. |
| `GET /api/stream/snapshot` | token | Current annotated frame as a JPEG. |
| `GET /docs` | token | Interactive API reference. |

API clients send the token as `Authorization: Bearer <token>`.

## Resilience built in

- **Playback supervisor** — a corrupt clip, decoder error or unplugged camera
  is logged and playback resumes with back-off, instead of freezing the video.
- **GPU out-of-memory** in detection frees the CUDA cache and continues.
- **USB camera unplugged** at start or while running: a demo clip plays
  and the camera is tried again every 10 s, then picked up by itself
  (CCTV cameras show "no signal" and reconnect, [CCTV.md](CCTV.md)).
- **Compiled models** (on a CPU) are checked when loaded; one that fails
  is skipped and the PyTorch model runs.
- **Slow CPU**: the activity model's looks are spaced by how long they
  take, so the video never stalls.
- **No changes at run time**: the monitor never installs or upgrades a
  package by itself (the detector library's auto-install is switched off).
- **Jetson memory**: the activity model reloads after freeing memory if
  the shared CPU/GPU memory was full.
- **Browser webcam** — if the viewer's camera stops sending for 8 s (tab
  closed, network lost) the server returns to the last clip.
- **Dashboard** — the live-update socket and the video stream reconnect on
  their own after a server restart or network blip.
- **Broadcaster** — one stalled viewer can no longer hold up, or crash,
  updates for everyone else.

## Batch processing

```bash
python scripts/process_video.py footage.mp4 --out runs/report
```

Same pipeline as the live view; writes an annotated MP4 plus JSON event log
and summary. In Docker:
`docker compose exec monitor python scripts/process_video.py data/samples/x.mp4`.
