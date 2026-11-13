# Running the demo on the RTX 3050 machine

The app uses the NVIDIA GPU automatically when one is available. Nothing in
the code or config needs editing.

## 1. Copy the project

Copy the `Human-Action-Behavior-Monitor` folder (or unzip
`HumanActionMonitor-GPU-package.zip`) to the GPU machine. Leave out `.venv`
— it is rebuilt there.

## 2. One-time setup (about 15 minutes, needs internet)

Double-click **`setup_gpu.bat`**. It does everything automatically:

- installs Python 3.11 if it is missing (through winget),
- installs the GPU build of PyTorch and every library (`requirements-gpu.txt`),
- downloads the CLIP model once,
- checks the NVIDIA driver and runs a GPU self-check.

The self-check must show the GPU, for example:

```
GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4.0 GB
App will use: cuda  |  activity look every 0.4s  |  detector 640px
```

The one thing a script cannot do is install the NVIDIA driver (it needs the
NVIDIA app or Windows Update and a reboot). If the self-check says
"App will use: cpu", update the driver and run `setup_gpu.bat` again.

## 3. Start it

Double-click **`run.bat`**. The browser opens http://localhost:8000.

- **Go live** (top right of the video panel) switches to the webcam.
- **Full screen** gives a clean viewing window for the client.
- Pick demo clips from the **Source** list.

## What the GPU changes

| | Laptop CPU | RTX 3050 |
|---|---|---|
| Activity model looks | every 1.5 s | every 0.4 s |
| Person detector size | 416 px | 640 px (finds smaller, more distant people) |
| Person / pose model | yolov8n-pose | yolov8s-pose, FP16 |
| Live accuracy, 15 demo clips | 11/15 | 15/15 (measured on a Kaggle T4) |
| Device | cpu | cuda |

Labels and subtitles should confirm noticeably faster, and more people in a
wide shot should be detected. Accuracy rules are unchanged.

## Before the client arrives

- Run `.venv\Scripts\python.exe scripts\gpu_check.py` once: it confirms the
  GPU and prints the time per model.
- Play two clips you will show (for example `demo-cooking-2`,
  `demo-using-computer`) and the webcam, and check the subtitle appears.
- Good, even lighting for the webcam; sit 1–3 m away, upper body in view.
