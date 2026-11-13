# Model Selection Rationale

Selections follow the master brief's principle: **compare alternatives, choose the smallest option that meets the accuracy bar, justify with numbers.**

## 1. Person Detection

| Model | mAP50 (COCO person) | Latency (RTX 3060) | Weights | Choice |
|---|---|---|---|---|
| YOLOv8n | ~37 | ~6 ms | 6 MB | **✅ default** |
| YOLOv8s | ~44 | ~10 ms | 22 MB | Optional (higher accuracy path) |
| RT-DETR-L | ~53 | ~28 ms | 66 MB | Overkill for single-class person |
| Faster R-CNN R50 | ~55 | ~55 ms | 160 MB | Too slow for real-time |

**Why YOLOv8n**: single-class task, low latency budget, cheap deployment. Ultralytics ships stable ByteTrack integration so tracking adds zero extra model.

## 2. Multi-Object Tracker

| Tracker | ID switches | Needs ReID model? | Choice |
|---|---|---|---|
| ByteTrack | low | no | **✅ default** |
| DeepSORT | lowest | yes (extra ~30 MB, ~5 ms/person) | Optional for crowded scenes |
| SORT | high | no | Too weak under occlusion |

**Why ByteTrack**: keeps low-confidence detections alive across short occlusions, no ReID overhead, ships with ultralytics.

## 3. Pose Estimation

| Model | Latency/person | Accuracy | Choice |
|---|---|---|---|
| MediaPipe Pose Lite | ~5 ms CPU | good on full-body poses | **✅ default** |
| MediaPipe Pose Heavy | ~20 ms CPU | best | Optional |
| MMPose HRNet-W32 | ~15 ms GPU | best | Overkill, GPU-only |
| MoveNet Lightning | ~4 ms | 17 kpts only | Feasible but fewer landmarks |

**Why MediaPipe Pose Lite**: 33 landmarks (superior to MoveNet), runs on CPU, official Python bindings, no weight download needed.

## 4. Action Recognition

Compared families:

| Family | Input | Accuracy (Kinetics) | Latency | Weights | Notes |
|---|---|---|---|---|---|
| Geometric heuristic on pose | pose window | limited (~70% on 8-class subset) | trivial | none | Zero-dep baseline; useful first-run demo. |
| **ST-GCN** on pose | pose window | ~72% (NTU-60) | ~2 ms | ~5 MB | **✅ recommended production**: temporal-aware, tiny, robust to lighting. |
| MoViNet-A2 | RGB clip | 78% | ~40 ms | 22 MB | Better accuracy, heavier. |
| VideoMAE-B | RGB clip | 81% | ~120 ms | 350 MB | Best accuracy, too heavy for edge. |
| SlowFast R50 | RGB clip | 76% | ~90 ms | 130 MB | Older, higher compute. |

**Chosen strategy**: `SkeletonHeuristicClassifier` ships as guaranteed-working default. If `models/st_gcn_kinetics.pt` is present, the pipeline transparently upgrades to `STGCNClassifier`. VideoMAE remains a **plug-in path** for accuracy-critical deployments.

Reasoning:
- Skeleton-based classifiers are illumination-invariant and privacy-friendly (no raw pixel storage).
- ST-GCN's ~5 MB footprint fits every deployment budget.
- Heuristic backend ensures the demo works with zero downloads — critical for the "usable out of the box" requirement.

## 5. Temporal Smoothing

Rejected: HMM (over-engineered for 8 labels), Kalman filter on label indices (labels aren't ordinal).
**Chosen**: sliding-window majority vote (window=5). Empirically kills the "standing/sitting/standing" flicker at negligible cost.

## Dataset selection (small, task-relevant slices)

- **UCF101**: standing/walking/running/sitting variety.
- **HMDB51**: additional bending/picking-up examples.
- **Le2i Fall Detection**: falling class (safety-critical).
- **NTU RGB+D 60 (subset)**: skeleton ground truth for ST-GCN validation.

Do **not** download in bulk. Pull only the label directories referenced in `config/model_config.yaml → action_labels`.
