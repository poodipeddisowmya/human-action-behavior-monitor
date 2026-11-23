# Human Action & Behavior Monitor: technical overview

What the system does, how it works, what it was measured to achieve, and how
to run, extend and demonstrate it. Written for decision-makers and engineers
alike: no prior knowledge of computer vision is assumed.

**Contents**
[1. In one paragraph](#1-in-one-paragraph) ·
[2. What it does](#2-what-it-does) ·
[3. What it recognises](#3-what-it-recognises) ·
[4. How it works](#4-how-it-works) ·
[5. The dashboard](#5-the-dashboard) ·
[6. Hardware and measured results](#6-hardware-and-measured-results) ·
[7. Making it yours](#7-making-it-yours-everything-that-can-be-changed-without-code) ·
[8. Face recognition and privacy](#8-face-recognition-and-privacy) ·
[9. Data, security and retention](#9-data-security-and-retention) ·
[10. Accuracy and limits](#10-accuracy-and-limits) ·
[11. Installing and running](#11-installing-and-running) ·
[12. Licences](#12-licences) ·
[13. The demo](#13-the-demo) ·
[14. Glossary](#14-glossary) ·
[15. Where to find what](#15-where-to-find-what)

---

## 1. In one paragraph

The monitor takes video from ordinary cameras (or recorded files), finds
every person in it, follows each one with a stable ID, recognises what they
are doing, names the people it has been taught to recognise, and keeps a log
of every activity with 2–3 photos as proof. Everything is shown live on a web
dashboard. It runs on a laptop, on a cloud GPU, or on a small NVIDIA Jetson
computer on site, where no video ever leaves the building.

## 2. What it does

| Capability | In practice |
|---|---|
| **People detection and tracking** | A box and a stable ID (P001, P002…) on every person, kept as they move. |
| **Activity recognition** | A label on each person: "cooking", "using phone", "walking"… updated several times a second. |
| **New activities from a description** | "Making tea" and "taking tablets" were added by describing them in words; no training videos were needed. |
| **Fall detection** | Someone falling is flagged within a second or two, as a red alert in the subtitle and the log. |
| **Face recognition** | Enrolled people are named: "Asha is making tea". Everyone else stays "Unknown". |
| **Name people on the spot** | An unknown person in view can be named from the dashboard with one click, and is recognised from then on. |
| **Subtitles** | A plain-English sentence describing the scene: "Two people in an office: one is using a computer." |
| **Activity log with proof** | Every activity is logged with who, what, when, how long, how sure, and photos of its start, middle and end. |
| **Video sources** | Bundled demo clips, uploaded videos, a USB camera, or the viewer's own browser webcam. |
| **Hardware comparison** | The same clips recorded on a laptop CPU, a cloud GPU and the Jetson, played side by side. |

## 3. What it recognises

**15 everyday activities** (from a video model trained on 400 human actions):
eating · drinking · cooking · cleaning · reading · writing · using phone ·
using computer · exercising · cycling · driving · dancing · swimming ·
brushing teeth · smoking

**3 activities added for this project** (on a GPU):

| Activity | How |
|---|---|
| Making tea | Described in words to a video transformer (X-CLIP) |
| Taking tablets | Described in words to a video transformer (X-CLIP) |
| Falling down | From the skeleton over time: upright, then on the ground within a second |

**10 body postures and movements** (from the skeleton, always available):
standing · sitting · walking · running · lying · kneeling · bending ·
crawling · push-up · falling

## 4. How it works

```
   Camera / video
        │
        ▼
 ① Find people ──────── a box and 17 body points (joints) per person, every frame
        │
        ▼
 ② Follow them ──────── the same ID for the same person, frame after frame
        │
        ▼
 ③ Look at each person with several specialists, at the same time:
        ├── skeleton rules ...... posture and movement (sitting, walking, falling)
        ├── activity model ...... what they are doing (cooking, using a phone)
        ├── object check ........ what is in their hands (cup, phone, laptop)
        ├── scene check ......... where they are (kitchen, office, street)
        ├── description model ... activities defined in words (making tea)
        └── face recognition .... who they are, if enrolled
        │
        ▼
 ④ Agree on one answer ─ the specialists vote; a label must hold for a moment
        │                 before it is shown, so it does not flicker
        ▼
 ⑤ Show and record ──── boxes and names on the video, a subtitle, and an
                         entry in the activity log with proof photos
```

Three design choices matter:

- **Several opinions, one answer.** No single model decides. The skeleton
  knows posture, the activity model knows motion, the object check knows
  what is in the hands. Combining them makes labels steadier and wrong
  labels rarer.
- **A restricted vocabulary.** The activity model knows 400 actions, but is
  only allowed to answer with the activities this project cares about, so
  unrelated actions ("juggling", "shot put") never appear.
- **Nothing blocks the video.** Each heavy model runs in its own background
  thread; the video plays at full speed and labels update as answers arrive.

The developer's view, with modules and threads:
[ARCHITECTURE.md](ARCHITECTURE.md). Every model and why it was chosen:
[MODEL_SELECTION.md](MODEL_SELECTION.md).

## 5. The dashboard

Open `http://<computer>:8000` in any browser. Three views:

**Overview** (the opening page) — the four features, each with real results
from this machine and a button that shows it working:
1. *Same video, three machines* — the hardware comparison.
2. *New activities, no retraining* — making tea, taking tablets, falling down,
   with the latest proof photo of each; one click plays a clip.
3. *Face recognition* — enrolled people and the photos that name them.
4. *Every activity logged, with proof photos* — the newest proof photos.

**Live monitor** — the working system:
- the video with boxes, names, activities and the subtitle;
- the **activity log**, filterable (all · new activities · known people ·
  alerts), each row with its proof photos;
- **Faces on screen** — unknown people in view, ready to be named;
- **Known people** — enrolled people: add, add photos, delete;
- **People seen recently**, with their history;
- the **Source** menu (clips grouped as New activities, Activities,
  Scenes) and **Upload**.

**CPU · T4 · Jetson** — the recorded comparison: the same clip on the three
machines side by side, frame-synced, with the measured figures.

## 6. Hardware and measured results

The same software runs on three kinds of hardware; it detects which one it is
on and sets itself up accordingly.

| | Laptop CPU | Cloud GPU (NVIDIA Tesla T4) | NVIDIA Jetson Orin Nano Super |
|---|---|---|---|
| What it is | Any modern laptop | A data-centre GPU (e.g. Kaggle, AWS) | A palm-sized edge computer, 8 GB, 25 W |
| Where video goes | Stays on the laptop | Sent to the cloud | Stays on site |
| Good for | Trying it out | Central processing of many cameras | Permanent on-site installation |

Measured on the same 29 labelled clips, 12 s each, as the screen showed them:

| | Laptop CPU | Tesla T4 | Jetson |
|---|---:|---:|---:|
| Activities clearly recognised | 1 / 29 | 21 / 29 | **22 / 29** |
| Correct label on screen (average) | 4% | 66% | **68%** |
| Wrong activity on screen | — | 0.5% | 0.8% |
| Playback speed | 18 fps | 24 fps | 22 fps |
| New activities clearly recognised (6 clips) | not run | 3 / 6 | **4 / 6** |

**What this means:**
- A GPU is needed for accuracy, not just speed. On a CPU the activity model
  gets one look every 1.5 s; on a GPU several a second, so the evidence
  agrees much sooner.
- The Jetson matches the cloud GPU while staying on site, which is the
  recommended production set-up for privacy and running cost.
- When the system is unsure it says less rather than something wrong:
  wrong labels are under 1% of the time on both GPUs.

## 7. Making it yours: everything that can be changed without code

| To | How |
|---|---|
| Recognise a person | Dashboard: type their name under their face in *Faces on screen*; or *Known people → Add person* with 2–5 photos; or from any proof photo. |
| Stop recognising a person | *Known people → delete*: removes every photo and record of their face. |
| Add an activity described in words | One entry in `config/model_config.yaml` under `open_vocab.activities`: a name and a few descriptions ([NEW_ACTIVITIES.md](NEW_ACTIVITIES.md)). |
| Add an activity the base model knows | One line in `src/action_recognition/target_vocabulary.py`. |
| Use a camera | A USB camera: dashboard *Live camera*, or start with `--source 0`. The viewer's own webcam: *Go live*. Network (RTSP/IP) cameras need a small source adapter ([ARCHITECTURE.md](ARCHITECTURE.md#extending-it)). |
| Analyse a recording | Dashboard *Upload*, or `scripts/process_video.py` for an annotated video and a report. |
| Protect access | Set `ACCESS_TOKEN` in `.env`. |
| Keep proof photos longer or shorter | `EVIDENCE_RETENTION_DAYS` in `.env` (default 30). |
| Turn features on or off | `.env`: `FACE_ID`, `OPEN_VOCAB`, `EVIDENCE`, `DEVICE`. |

Every threshold in `config/model_config.yaml` carries a comment saying what
it was measured against, so changes are informed rather than guesses.

## 8. Face recognition and privacy

**How a person gets a name.** A background check looks at each person's
face a few times a second. A face counts only when it is clear (sharp,
large enough, facing the camera) and sits at that person's own head, so a
face in a crowd is never taken for someone else's. A person is named when 2
of 3 clear looks agree with an enrolled person, and stays Unknown otherwise.

**Measured:** the same person scored 0.45–0.96 similarity across a clip;
different people never above 0.27, against a bar of 0.363. Two people
enrolled from one photo each were named within 1.5 s; four other people
stayed Unknown on all 74 looks.

**Privacy by design:**
- Only enrolled people are ever named. Faces of people who are not named are
  held in memory only while they are in view, and never stored.
- Face data stays on the device (`data/faces/` and the local database).
- Deleting a person removes all of their face data at once.
- Enrolment refuses unclear photos, photos with several faces, and faces that
  match another person or do not match the person's other photos.

**The law.** A face is biometric personal data. Under India's Digital
Personal Data Protection Act 2023 (and the GDPR in Europe), people must agree
to be enrolled, know what it is for, and be able to have it deleted. The
dashboard reminds the operator at every enrolment. Details:
[FACE_RECOGNITION.md](FACE_RECOGNITION.md).

## 9. Data, security and retention

| Data | Where | How long |
|---|---|---|
| Activity log (who, what, when) | Local database (`logs/monitor.db`; PostgreSQL possible) | Until deleted |
| Proof photos | `data/evidence/` | 30 days by default, then deleted automatically |
| Enrolled faces | `data/faces/` and the database | Until the person is deleted |
| Video | Not recorded; processed live | — |

- **Access:** set `ACCESS_TOKEN` on any machine others can reach; every page
  and API call then needs it. Without it, anyone on the network can open the
  dashboard.
- **Network:** on a Jetson or laptop, nothing is sent outside. Model files
  are downloaded once at installation.
- **Integration:** everything on the dashboard is available through a
  documented API ([API.md](API.md)), and live at `/docs` on the server, for
  connecting alarms, reports or other systems.

## 10. Accuracy and limits

What to expect, stated plainly:

- **Typical accuracy:** on a GPU, 21–22 of 29 test clips were clearly
  recognised and the correct label was on screen about two-thirds of the
  time; wrong labels under 1%. The rest of the time the system shows a
  posture ("standing") or nothing specific rather than a wrong activity.
- **The first label** appears about 2 s after a person comes into view.
- **Camera placement matters most:** 1.5–3 m away, the upper body and hands
  visible, light in front of the person. Faces must be visible and at least
  ~40 pixels wide to be recognised.
- **Crowds:** people hidden behind others are tracked but labelled less
  reliably.
- **Test footage:** these are public clips. Accuracy on the client's own
  cameras should be measured on site.

**The biggest improvement available** is fine-tuning the activity model on a
few hundred labelled clips from the client's own site and cameras. The
project is built for it (the evaluation scripts measure before and after).

## 11. Installing and running

Short version (full steps in the [README](../README.md)):

| Machine | Install | Start |
|---|---|---|
| Windows laptop | `setup.bat` | `run.bat` |
| Linux / macOS | `pip install -r requirements.txt` in a virtual environment | `python scripts/run_pipeline.py` |
| NVIDIA Jetson | `./setup_jetson.sh` ([JETSON.md](JETSON.md)) | `./run_jetson.sh` |
| Cloud GPU (Kaggle) | [KAGGLE_DEMO.md](KAGGLE_DEMO.md) | one notebook cell |
| Server (Docker, GPU) | [DEPLOYMENT.md](DEPLOYMENT.md) | `docker compose up -d --build` |

Before anything else: **install Git LFS and clone with git** (not "Download
ZIP"); the model files and videos are stored with Git LFS.
`python scripts/verify_install.py` checks the installation and prints the
fix for anything missing. The demo clips for the new activities are
downloaded separately (their licences do not allow re-sharing them):
`python scripts/fetch_new_activity_clips.py`.

## 12. Licences

Third-party models and clips each carry their own licence
([MODEL_SELECTION.md](MODEL_SELECTION.md),
[data/showcase/CREDITS.md](../data/showcase/CREDITS.md)). Two need a decision
before a commercial product:

- **YOLOv8 (Ultralytics) is AGPL-3.0.** A closed-source product needs an
  Ultralytics Enterprise licence, or a different detector.
- **The VideoMAE activity-model weights used on GPUs are non-commercial
  (CC BY-NC 4.0).** For commercial use: the S3D model (BSD licence, already
  supported, used on CPUs), or a model fine-tuned on the client's own footage.

Everything else (ByteTrack, CLIP, X-CLIP, SmolVLM2, the face models) allows
commercial use.

## 13. The demo

About 10 minutes, all from the dashboard's **Overview**, in the order of its
four cards. The step-by-step script, with what to click and what to say:
[CLIENT_DEMO.md](CLIENT_DEMO.md).

| Minutes | Show | The point |
|---|---|---|
| 1 | Overview | One page: what it does, with real numbers from this machine. |
| 2 | Card 1 → comparison | The same clips on a laptop, a cloud GPU and the Jetson: the GPU makes it accurate; the Jetson matches the cloud on site. |
| 2 | Card 2 → making tea, taking tablets, falling down | New activities added by describing them, no retraining; a fall raises an alert. |
| 3 | Card 3 → Classroom clip → Faces on screen | Name someone with one click; the box, subtitle and log name them at once. Strangers stay Unknown. |
| 2 | Card 4 → activity log | Every activity with who, when, and photo proof; filters for alerts and known people. |

## 14. Glossary

| Term | Meaning |
|---|---|
| Tracking ID | The number a person keeps while in view (P001). It restarts when the video changes. |
| Keypoints / skeleton | 17 body points (nose, shoulders, knees…) found on every person. |
| Kinetics-400 | A public collection of 400 human actions the activity model learned from. |
| Zero-shot | Recognising something from a description, without training examples of it. |
| Enrolment | Teaching the system a person's face, with their consent. |
| Proof photos | 2–3 photos of the person taken during a logged activity. |
| GPU | A graphics processor; runs the vision models many times faster than a CPU. |
| Jetson | NVIDIA's small, low-power computer with a GPU, for on-site use. |
| TensorRT | NVIDIA's optimiser that makes the detector faster on a Jetson. |
| fps | Frames per second: how smoothly the video plays. |

## 15. Where to find what

| Document | For |
|---|---|
| [README](../README.md) | Installing and starting |
| [CLIENT_DEMO.md](CLIENT_DEMO.md) | The demo, step by step |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Modules, threads, extension points |
| [MODEL_SELECTION.md](MODEL_SELECTION.md) | Every model, why, and its licence |
| [NEW_ACTIVITIES.md](NEW_ACTIVITIES.md) | Making tea, taking tablets, falls; adding more |
| [FACE_RECOGNITION.md](FACE_RECOGNITION.md) | Face recognition, enrolment, privacy |
| [JETSON.md](JETSON.md) | The Jetson: set-up, tuning, the comparison |
| [KAGGLE_DEMO.md](KAGGLE_DEMO.md), [GPU_DEMO.md](GPU_DEMO.md) | Running on a cloud or local GPU |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Production server (Docker) |
| [API.md](API.md) | Connecting other systems |
| [scripts/README.md](../scripts/README.md) | Every script |
