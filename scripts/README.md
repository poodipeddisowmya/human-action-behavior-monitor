# Scripts

Run them from the project folder with the project's Python
(`.venv\Scripts\python.exe` on Windows, `.venv/bin/python` elsewhere), e.g.
`python scripts/verify_install.py`. Each script's first lines explain it in
full; `--help` lists its options.

## Everyday use

| Script | What it does |
|---|---|
| `verify_install.py` | Checks a new installation and prints the fix for anything missing. Run it first. |
| `run_pipeline.py` | Starts the monitor and dashboard (`--source <clip>`, `--source 0` for a camera). |
| `fetch_new_activity_clips.py` | Downloads the demo clips not in the repository from their sources: making tea, taking tablets, falling down, and the Indian office set. |
| `record_live.py` | Records what the dashboard shows (boxes, names, subtitles) to MP4, from whatever is playing. |
| `test_camera.py` | Plays a video file as a network camera, to rehearse the CCTV path without one ([CCTV.md](../docs/CCTV.md)). |
| `download_models.py` | Downloads model weights again, if any are missing. |
| `process_video.py` | Processes a video file offline into an annotated video and an activity report. |
| `live_window.py` | Watches a running monitor in a desktop window instead of a browser. |
| `smoke_test.py` | Checks a deployed server: is it up, recognising, and real time? |
| `gpu_check.py` | Is the GPU used, and how fast is each model on it? |
| `install_autostart.ps1`, `keep_running.ps1`, `uninstall_autostart.ps1` | Windows: start the monitor at log-on and restart it if it stops. |

## GPU, Jetson and Kaggle

| Script | What it does |
|---|---|
| `build_trt_engines.py` | Builds the TensorRT detector engine (NVIDIA GPU or Jetson; once per device). |
| `kaggle_launch.py` | Runs the monitor in a Kaggle GPU notebook with a public link. |
| `kaggle_auto.py`, `package_for_kaggle.py` | Packages the project and runs the Kaggle GPU tests from this PC. |

## The CPU / T4 / Jetson comparison

| Script | What it does |
|---|---|
| `record_showcase.py` | Records the comparison clip set on this machine. |
| `install_showcase.py` | Installs a recording into the dashboard's comparison tab. |
| `add_showcase_side.py` | Adds another device (e.g. a Jetson) as a column. |
| `update_showcase_gpu.py` | Replaces the GPU column with a new recording. |
| `set_baseline.py` | Uses a local recording as the left-hand (CPU) column. |
| `compare_devices.py` | Runs the same clips here and compares with the recorded devices. |
| `build_comparison.py`, `build_showcase.py` | Before/after videos and an offline page from a recording. |

## Accuracy and calibration

| Script | What it does |
|---|---|
| `eval_offline.py` | Accuracy on the labelled demo clips (`--realtime` for live-equivalent numbers). |
| `live_accuracy.py` | Accuracy as a viewer sees it, against a running server. |
| `eval_open_vocab.py` | Measures the text-described activities (X-CLIP) on clips. |
| `validate_demo_set.py`, `check_always_labels.py` | Every demo clip gives its expected label; every person always has one. |
| `startup_probe.py`, `profile_pipeline.py`, `benchmark.py` | Time to the first label; where frame time goes; FPS, memory, GPU. |

## Developer experiments

Kept for reference: their measurements set several of the calibrated values
in `config/model_config.yaml`.

`bench_classifiers.py`, `bench_s3d_configs.py`, `diag_pose_rules.py`,
`diag_sitting.py`, `eval_candidate_clips.py`, `exp_crop_vs_fullframe.py`,
`exp_squash_vs_letterbox.py`, `measure_walk_speed.py`,
`fetch_pexels_clips.py`, `get_sample_video.py`, `dedupe_samples.py`.
