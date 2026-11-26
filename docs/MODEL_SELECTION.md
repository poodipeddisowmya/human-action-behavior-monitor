# Models used, and why

Every model is pretrained and publicly available; none was trained on
client data. All licences allow commercial use except where noted.

| Job | Model | Runs on | Licence | Why this one |
|---|---|---|---|---|
| Find people and their skeleton | YOLOv8n-pose (CPU), YOLOv8s-pose (GPU; TensorRT on a Jetson) | every frame | AGPL-3.0 (Ultralytics) | People and 17 body keypoints in one pass; the skeleton feeds posture, falls and face matching at no extra cost. |
| Keep IDs stable | ByteTrack | every frame | MIT | Survives short occlusions without a separate re-identification model. |
| Posture and motion | Skeleton rules (`action_classifier.py`) | every frame | this project | Sitting, standing, walking, lying… read directly from the joints: instant and explainable. |
| Everyday activities | VideoMAE-base, Kinetics-400 (GPU); S3D, Kinetics-400 (CPU) | several looks a second | CC BY-NC 4.0 (VideoMAE weights); BSD (S3D, torchvision) | Strong video models for 400 actions, restricted here to the 15 target activities so unrelated classes never compete. |
| Objects in hand | YOLOv8n (COCO) | with the activity model | AGPL-3.0 | A cup, phone or laptop near the hands confirms drinking, using a phone, using a computer. |
| Place, and sitting/standing when legs are hidden | CLIP ViT-B/32 | every few seconds | MIT | Zero-shot: a place is one line of text. |
| Activities never trained on | X-CLIP base/16, zero-shot | once a second (GPU) | MIT | A video transformer that scores video against text: "making tea", "taking tablets" are a description, not a training set. |
| Falls | Fall rules (`fall_detector.py`) | every frame | this project | A fall is a change of state (upright → on the ground within ~1 s); 9 of 9 real falls, no false alarms on 43 other clips. |
| Find faces | YuNet (OpenCV Zoo) | CPU, a few looks a second | MIT | Small and fast; runs in OpenCV with no extra dependency. |
| Recognise faces | SFace (OpenCV Zoo) | CPU | Apache-2.0 | Accurate enough at the published 0.363 bar; InsightFace was rejected as non-commercial only. |
| Detailed subtitles | SmolVLM2-500M-Video-Instruct | T4-class GPU | Apache-2.0 | Describes the scene from the frames, grounded in the detected labels. |

## Licences to check before a commercial deployment

- **Ultralytics YOLOv8 is AGPL-3.0.** Using it in a product that is not
  itself open source needs an Ultralytics Enterprise licence, or a switch
  to a permissively licensed detector.
- **The VideoMAE Kinetics weights are CC BY-NC 4.0** (non-commercial). On a
  commercial deployment use the S3D backend (`deep_backend: s3d`, BSD), and
  train your own activities on it (`scripts/train_activities.py --backend
  s3d`, [TRAINING.md](TRAINING.md)).
- Everything else above allows commercial use.

## Measured

On the same 29 labelled clips, 12 s each, as the screen showed them:

| | Laptop CPU | Tesla T4 | Jetson Orin Nano Super |
|---|---:|---:|---:|
| Clips clearly recognised | 1/29 | 21/29 | 22/29 |
| Correct label on screen | 4% | 66% | 68% |
| Playback speed | 18 fps | 24 fps | 22 fps |

The GPU matters for accuracy, not only speed: the activity model gets
several looks a second instead of one every 1.5 s, so its votes agree
sooner. Details: [JETSON.md](JETSON.md), [NEW_ACTIVITIES.md](NEW_ACTIVITIES.md),
[FACE_RECOGNITION.md](FACE_RECOGNITION.md).
