# Production deployment on a GPU server

## Option A — Docker (recommended)

Host requirements: Linux, NVIDIA driver, [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html),
Docker with Compose. Any NVIDIA GPU with ≥4 GB works (RTX 3050 upwards, T4, L4, A10).

```bash
echo "ACCESS_TOKEN=$(openssl rand -hex 16)" > .env
docker compose up -d --build
docker compose logs -f          # "pipeline started on cuda (NVIDIA ...)"
```

Open `http://<server>:8000/?token=<ACCESS_TOKEN>`. Put it behind HTTPS (a
reverse proxy such as Caddy or nginx, or a Cloudflare tunnel) if viewers
will use **Go live** with their own webcam — browsers allow cameras only on
https or localhost.

The event database, logs and uploaded clips are on named volumes and survive
rebuilds. Docker restarts the container if it exits, and the image's
`HEALTHCHECK` marks it unhealthy if the server stops answering.

## Option B — Windows machine with an NVIDIA GPU

`setup_gpu.bat`, then `run.bat`. See [GPU_DEMO.md](GPU_DEMO.md). Add
`ACCESS_TOKEN=...` to `.env` if the machine is reachable from a network you
don't control.

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

On a GPU, the `gpu:` block of `model_config.yaml` applies automatically:
full-resolution detection, FP16 detector, activity model 2.5× a second.

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
