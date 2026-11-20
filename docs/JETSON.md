# Running on an NVIDIA Jetson

Tested on a **Jetson Orin Nano Super Developer Kit** (8 GB, JetPack 6.2.1,
L4T R36.4.7, CUDA 12.6, TensorRT 10.3, Python 3.10, power mode `MAXN_SUPER`),
2026-11-16.

## Setup — once, no sudo needed

```bash
./setup_jetson.sh
```

It:

1. installs `git-lfs` into `~/.local/bin` (official release, checksum-checked)
   and pulls the weights and demo clips. A clone without LFS leaves 132-byte
   pointer files in their place, and nothing works;
2. creates `.venv` with `virtualenv`, because JetPack has no `python3-venv`;
3. installs `requirements-jetson.txt`: the **CUDA builds of torch 2.8 /
   torchvision 0.23 from NVIDIA's Jetson AI Lab index** (PyPI's aarch64
   wheels run on the CPU only), ffmpeg via `imageio-ffmpeg`, and links
   JetPack's TensorRT into the venv;
4. downloads CLIP, VideoMAE and SmolVLM2;
5. builds the TensorRT engine for the pose detector (~13 min, once);
6. runs `scripts/verify_install.py` and `scripts/gpu_check.py`.

Optional, for the fastest CPU-side processing (needs your password):
`sudo jetson_clocks` locks CPU, GPU and memory clocks at maximum; without it
the CPU cores drop to 729 MHz between bursts.

## Run

```bash
./run_jetson.sh                       # dashboard on http://<jetson-ip>:8000
./run_jetson.sh --source 0            # USB camera
.venv/bin/python scripts/live_window.py --fullscreen   # live view on the Jetson's screen
```

### Live camera

Tested with a Logitech C270: 30 fps live, ~33 ms per frame to process.

- Plug the camera in **before** starting the server: cameras are listed once,
  at start-up. Then pick it under *Source*, or start with `--source 0`.
- `scripts/live_window.py` shows the live video (boxes, labels, subtitle) in
  a plain window, ~75 MB, against ~600 MB for a browser that the models need.
  `f` full screen, `q` quit.
- Watching from another computer, open `http://<jetson-ip>:8000` in its
  browser rather than viewing the Jetson's screen over remote desktop
  (RustDesk, VNC): re-encoding the screen makes live video lag and stutter.
- In dim light the C270 lengthens its exposure and drops to 15 or 7.5 fps.
  More light fixes it; so does turning off its exposure priority
  (`v4l2-ctl -c exposure_auto_priority=0`, from `v4l-utils`), at the cost of
  a darker picture. The setting lasts until the camera is unplugged.
- Labels need the person 1.5–3 m away with hands in view and the light in
  front: a head-and-shoulders close-up, or a bright window behind, leaves the
  activity model guessing at low confidence.

## Results: laptop CPU vs Tesla T4 vs Jetson

Same 29 labelled clips, 12 s each, scored on what the screen showed
(`scripts/build_showcase.py` rules). Laptop CPU and T4 from
`data/showcase/showcase.json`; Jetson recorded here.

| | Laptop CPU | Tesla T4 | Jetson, T4 settings | Jetson, optimised | Jetson, per-person fix | Jetson, new activities | **Jetson, final** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clearly recognised (correct label ≥ half the time) | 1/29 | 23/29 | 23/29 | 22/29 | 24/29 | 23/29 | **22/29** |
| Correct label on screen, average | 4% | 70% | 72% | 69% | 73% | 70% | **68%** |
| Wrong activity on screen, average | 0.3% | 0.9% | 1.2% | 2.0% | 0.6% | 0.9% | **0.8%** |
| Correct label first appeared (median) | 1.4 s | 1.2 s | 1.8 s | 1.0 s | 1.3 s | 1.9 s | 2.1 s |
| Playback FPS (median; capped at each clip's own rate) | 17.5 | 24.0 | 22.8 | 22.8 | 22.6 | 22.4 | 22.2 |
| Failed model looks during the run | – | – | 2 | 0 | 0 | 3 of 1,048 | 2 of 1,041 |

*Per-person fix* is *optimised* plus the per-person label fix below
(2026-11-17). *Current* adds the new activities (making tea, taking tablets,
falling down; [NEW_ACTIVITIES.md](NEW_ACTIVITIES.md)) and is the Jetson side
of the dashboard's comparison tab (2026-11-18). Its accuracy is within the
run-to-run spread (the two clips that moved, `demo-eating` and
`hd-swimming-2`, have swung 0-56% between earlier runs too); none of the 32
clips is ever labelled with a new activity. The one real cost: X-CLIP shares
the Orin Nano's GPU with the activity model, and the first correct label
comes about 0.6 s later (measured in two runs). `OPEN_VOCAB=0` removes it,
and the new activities with it. The laptop CPU's low wrong-activity figure is
because it rarely shows any activity at all.

*Final* (2026-11-19) adds the strong-look rule for the new activities,
X-CLIP without its text encoder (120 MB less GPU memory) and face
recognition (CPU), and is the Jetson side of the comparison tab. The 1-2
clip moves between the last three columns are run-to-run: clips such as
`demo-cycling` and `demo-basic-walk` sit right at the 50% "clearly" bar.

**The T4 with the same code** (re-recorded on Kaggle 2026-11-18 and again
2026-11-19, the GPU side of the comparison tab now): 23/29 then 21/29
clearly recognised, 67% / 65% on screen, 0.3% / 0.5% wrong, 23.5 FPS. Its
per-clip first-label times are unchanged from the original T4 recording
(cooking 1.2 s, using computer 1.3 s); the median moved to ~2 s only
because a different set of clips showed the label at all. So the ~0.6 s
cost above is the Orin Nano's small GPU, not the new layers.

- **One person's activity is no longer copied onto everyone.** With several
  people in view, a person the activity model had not confirmed anything
  for borrowed the scene-wide verdict, which pools everyone's looks, and
  kept it even after their own looks found nothing; it also overrode their
  walking or running. On a live webcam that painted one weak guess
  ("smoking", "using phone 54%") on three people, one of them just
  standing. Now a person borrows the scene verdict only until their own
  first look, and measured walking/running beats a borrowed label
  (`src/action_recognition/videomae_classifier.py`). One-person scenes are
  unchanged. Over the 29 clips: wrong activity on screen 2.0% → 0.6%,
  `demo-basic-run` 37% → 81% (a borrowed "dancing" gone), no clip worse.
- **The Jetson matches the T4 on accuracy.** The ±1 clip differences are
  run-to-run variation on borderline clips: re-recording `hd-swimming-2`
  with *all* Jetson tuning off gave 0% swimming twice, against 62% in the
  first run (the model mostly sees "jumping into pool", outside the
  vocabulary).
- **Optimised, labels appear sooner than on the T4**: paired over the 23
  clips where both show the correct label, 0.46 s sooner than the Jetson
  with T4 settings (faster on 18 of 23) and 0.24 s sooner than the T4
  (faster on 18 of 23).
- The laptop-CPU column runs the same GPU-sized models, which a CPU cannot
  keep up with; it shows why the GPU matters, not the CPU profile's best.

## The Jetson profile (`jetson:` in `config/model_config.yaml`)

Applied on a Jetson on top of the `gpu:` profile. `JETSON_OPTIMISED=0`
turns it off and reproduces the T4's configuration exactly.

| Setting | Measured on the Orin Nano | Check |
|---|---|---|
| `detector_engine` — yolov8s-pose as an FP16 TensorRT engine (dynamic 640/960 px) | GPU time 26.8 → 11.9 ms at 640 px, 28.6 → 19.5 ms at 960 px | Same people per frame; boxes 0.96–0.995 IoU against PyTorch |
| `deep_half` — VideoMAE in FP16 (softmax in FP32) | 202 → 75 ms per look | 108 looks, 36 clips: same winning activity 108/108, confidence within 0.003 |
| `clip_half` — CLIP loaded in FP16 | Same speed, half the 600 MB of weights | – |
| `vlm_captions: false` | SmolVLM2 does not fit beside a desktop session (sentences and person detections failed) | Template subtitles; labels unaffected. `VLM_CAPTIONS=1` on a headless board |

GPU busy (tegrastats `GR3D_FREQ`) fell from 50% to 44% on average for the
same work. The remaining per-frame cost is CPU-side Python (frame resize,
box and keypoint decoding: 12–24 ms), which `jetson_clocks` speeds up.

**Looking more often does not help.** With VideoMAE at 75 ms the GPU has
room for more activity looks, so `deep_min_interval_s` (the gap between
looks, 0.4 s) was tried shorter, 2026-11-17:

| Gap | Looks/s, one person | FPS | 29 clips: clearly recognised | on screen | first shown (median) |
|---|---:|---:|---:|---:|---:|
| 0.4 s (kept) | 1.8 | 22.8 | 22/29 | 69% | 1.0 s |
| 0.2 s | 2.1–2.2 | 22.8 | 21/29 | 70% | 1.1 s |
| 0.1 s | 2.1–2.3 | not recorded on all 29 clips | | | |

Looks/s and the 0.1 s row come from a sweep over a few clips
(smoking, dancing, two crowd scenes). Each person submits a clip only every 8 frames, which caps the rate near
2.2 looks/s whatever the gap; the 0.2 s run differed from 0.4 s only by
run-to-run noise (three clips moved, in both directions).

## Jetson-specific fixes in the code

| Issue | What happened | Fix |
|---|---|---|
| cuDNN autotune | `NVML_SUCCESS == r INTERNAL ASSERT FAILED` on the first S3D/VideoMAE look | `config/settings.py` disables `cudnn.benchmark` on a Jetson |
| Shared memory | CPU and GPU share 8 GB, and CUDA allocations do not evict the page cache: models fail to load (`NvMapMemAlloc error 12`) while `free` shows GBs "available" | `run_jetson.sh` / `record_showcase.py` touch a 2 GB buffer first, which evicts it (no sudo) |
| Idle CUDA contexts | The batch parent of `process_video.py` held ~300 MB of GPU memory while only waiting | It imports the app only in the per-clip worker |

Memory: with the desktop and the Claude app open, ~4 GB is left for the
monitor, which needs ~3 GB optimised. Close browsers and the software
updaters before a run; a headless board has room for SmolVLM2 too.

## Reproduce the comparison

```bash
python scripts/record_showcase.py --out dist/jetson-current             # ~19 min
JETSON_OPTIMISED=0 VLM_CAPTIONS=0 \
  python scripts/record_showcase.py --out dist/jetson-showcase          # ~22 min
python scripts/compare_devices.py "dist/jetson-current=Jetson (current)" \
                                  "dist/jetson-showcase=Jetson (T4 settings)"
```

`record_showcase.py` records the same clips as `kaggle/gpu_showcase.py` and
lists, per clip, any model that failed to load or failed a look.

## In the dashboard

The **CPU vs GPU** tab (`http://<host>:8000/#compare`) shows all three side
by side: laptop CPU, Tesla T4 and Jetson, the same clip playing in lock-step
on each, with the totals above. The Jetson side is the current recording,
added with:

```bash
python scripts/add_showcase_side.py dist/jetson-current --key jetson \
    --tag "EDGE GPU" --sub "Jetson Orin Nano Super (8 GB)" --short Jetson
```

It copies the videos to `data/showcase/jetson/` and the results into
`data/showcase/showcase.json` (`extra_sides`); re-running it replaces that
side. Any other device recorded with `record_showcase.py` can be added the
same way under its own key.
