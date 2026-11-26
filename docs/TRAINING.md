# Teaching it a new activity

The monitor recognises 15 everyday activities out of the box, plus making
tea, taking tablets and falling down. For anything else your site needs
(someone asleep at a desk, a fight, a person lying in a corridor, a
restricted door being opened), you train it on your own clips. No machine
learning knowledge is needed: you sort clips into folders and run one
command. Training runs on the machine that runs the monitor: a Jetson,
a PC with an NVIDIA GPU, or a CPU-only PC (see [where to train](#where-to-train)).

| You want | Do this | Needs |
|---|---|---|
| An activity the base model already knows (one of Kinetics' 400) | One line in `src/action_recognition/target_vocabulary.py` | nothing else |
| An activity you can describe in words | One entry under `open_vocab` in `config/model_config.yaml` ([NEW_ACTIVITIES.md](NEW_ACTIVITIES.md)) | a GPU |
| **Anything else** | **Train it on clips: this page** | 8+ clips, ideally 20-50 |
| A built-in activity, sharper on your cameras | Train it too: name the folder like the built-in one (`using phone/`), with clips from your cameras | 20-50 clips |

## 1. Collect clips

A clip is a short video (2-30 s) in which the activity is visible
**throughout**. Cut them from your recordings, or film them.

- **From your own cameras.** Same angle, distance and lighting as in use:
  clips from the actual cameras are worth far more than stock footage.
- **Varied.** Different people, clothes, times of day, positions in the
  picture. Ten clips of one person in one chair teach the model that person
  and that chair.
- **At least 8 clips per activity; 20-50 is better.** More, varied clips is
  the single biggest improvement.
- **The person clearly visible**, at least about a quarter of the picture's
  height. Clips in which nobody is detected are skipped.
- **Only the activity.** Trim off the walking-in and the walking-away: they
  are not the activity, and teach the wrong thing.

To cut clips from a long recording, any video editor works, or ffmpeg:

```bash
ffmpeg -ss 00:12:30 -to 00:12:45 -i camera1-2026-11-20.mp4 -c copy "data/training/sleeping at desk/cam1-001.mp4"
```

## 2. Put them in folders

```
data/training/
├── sleeping at desk/        the folder's name is the activity's name on screen
│   ├── cam1-001.mp4
│   ├── cam3-002.mp4
│   ├── ...
│   └── phrase.txt           optional: subtitle wording, e.g. "sleeping at their desk"
├── fighting/                as many activities as you like, one folder each
│   └── ...
└── other/                   recommended: ordinary activity on the same cameras
    └── ...
```

**`other/`** teaches the model what the new activity is **not**: people
working at their desks, walking past, talking, on the phone, eating, on the
same cameras. The bundled demo clips (`data/samples/`) are added to it
automatically. The more your `other/` looks like your cameras on a normal
day, the fewer false alarms. If one kind of ordinary activity triggers the
new one, add clips of it to `other/` and retrain.

Folder names: letters, digits, spaces, `'` and `-`; `sleeping_at_desk`
reads as "sleeping at desk". `data/training/` is never committed to git
(it holds your site's footage).

## 3. Check and train

On a Jetson, stop the monitor first (the activity model needs the GPU's
memory), then:

```bash
.venv/bin/python scripts/train_activities.py --check     # are the clips usable?
.venv/bin/python scripts/train_activities.py             # train, measure, install
```

(`.venv\Scripts\python.exe` on Windows.) `--check` lists each activity with
its number of usable clips and why any clip was skipped (damaged, too short).

Training prints a report like this one, from the example in this repository
(people asleep at desks, against ordinary activity; Jetson, VideoMAE):

```
  Held-out clips at threshold 0.80 (live rule: 3 of the last 4 looks):
    sleeping at desk             recognised in  18/26  clips
    other (ordinary activity)    false alarm in   0/99 clips

  Other thresholds (--threshold T to choose one; higher = fewer false alarms):
    threshold  recognised  false alarms
      0.60      20/26         3/99
      0.70      19/26         2/99
      0.80      18/26         0/99  <- chosen
      0.90      16/26         0/99
```

**Held-out** means each clip was judged by a model trained without it, so
these numbers predict how it does on new video, not on what it memorised.
The threshold is chosen automatically: the lowest that raises no false
alarm on the `other` clips. If missing some is worse for you than an
occasional false alarm, pick a lower one from the table (`--threshold 0.7`).
What to do with the report:

| Report says | Do this |
|---|---|
| Recognised in under 70% of clips | More clips, more varied (people, angles, light) |
| False alarms on `other` clips | Add clips like those to `other/` and retrain |
| Mistaken for another activity | The two look alike on camera: more clips of both, or merge them |

Then restart the monitor. The log says
`[custom] trained activities (videomae): sleeping at desk`, and the activity
appears like any other: on the video, in the subtitle ("A person is sleeping
at their desk"), in the activity log with photos, in the database and the
API.

## 4. Which activity model

The layer is trained on top of the activity model the monitor uses, and
must match it:

| Machine | Activity model | Train with |
|---|---|---|
| Jetson, NVIDIA GPU | VideoMAE (most accurate; weights **non-commercial**) | `--backend videomae` (the default there) |
| CPU only, or commercial use | S3D (BSD licence, commercial use allowed) | `--backend s3d` |
| Both, to switch freely | | `--backend both` |

On the example, with no false alarm on the 99 ordinary clips, VideoMAE
recognised 18 of 26 held-out clips and S3D 12 (S3D: 20 of 26 if 2 false
alarms in 99 are acceptable, `--threshold 0.7`). S3D benefits most from
more clips from your own cameras.

For a commercial product, use S3D everywhere (`deep_backend: s3d` in
`config/model_config.yaml`, both the `model:` and `gpu:` sections) and train
with `--backend s3d`. The monitor tells you in its log if it finds a model
trained for the other backbone.

### Where to train

Nothing in training is specific to the Jetson: the same command works on
any machine that runs the monitor (Linux or Windows; a Jetson, a PC with
an NVIDIA GPU, or any edge board or PC without one), and trains for the
backbone that machine uses.

| Machine | Backbone it uses | Training |
|---|---|---|
| Jetson Orin | VideoMAE | minutes; stop the monitor first (shared memory) |
| PC with an NVIDIA GPU | VideoMAE | minutes; the monitor can keep running if the GPU has room |
| No NVIDIA GPU (Raspberry Pi 5, RK3588, mini PC, laptop) | S3D | works; uses the models compiled for the CPU by setup (about 0.5 s per look on an ARM board, so 50 clips take roughly 10-15 minutes) |

The trained files (`models/custom_activities/<backbone>.npz` and `.json`)
are small and **portable**: train once, then copy them to every box that
uses the same backbone, or commit them. A model trained on a Jetson
(VideoMAE in half precision) and features computed the PC way (full
precision, another person detector) gave the same verdict on every clip
we compared. To be sure on a new machine, retrain there: features are
cached, so it costs one pass over the clips.

## 5. Updating and undoing

- **Retrain** whenever you add clips: same command. Features already
  computed are cached (`data/training/.cache/`), so only new clips cost time.
- **Every install keeps the previous model** in
  `models/custom_activities/backup/`: copy it back to undo.
- **Remove all trained activities:** `scripts/train_activities.py --remove`.
- **Switch them off** without removing: `CUSTOM_ACTIVITIES=0` in `.env`.
- Live tuning in `config/model_config.yaml`, `custom_activities:`
  `threshold` (higher = fewer false alarms, fewer detections), `min_hits`,
  `hold_s`.

## Measured: the example on the Jetson (2026-11-26)

*Sleeping at desk*, trained on 26 stock clips of people asleep at desks
against 99 ordinary ones (44 in `other/`, including writing heads-down,
studying, thinking with a hand on the chin; plus the demo clips), VideoMAE:

| Test | Result |
|---|---|
| Held-out clips during training (cross-validation) | 18 of 26 recognised, 0 of 99 ordinary clips raised it |
| **Live, 8 sleeping clips never used in training** | **7 of 8** recognised, first shown 1.5-2.3 s in; the miss read as *using computer* |
| **Live, 9 ordinary clips never used in training** (5 of them look-alikes) | **0 false alarms** |
| Live, 16 demo clips (classrooms, office, writing, smoking...) | never shown; their usual activities unchanged |

Two things this run fixed, now part of the software: requiring 3 of 4
looks (with 2 of 3, head-down writing and a slumped smoker triggered it),
and choosing the model's regularisation by recall at zero false alarms.

## How it works

The activity model turns every 16-frame look at a person into a summary
vector of what it sees. Training learns a small layer that reads those
vectors: one score per new activity, plus "other". The activity model
itself is not changed, so everything it already recognises keeps working
exactly as before. The training clips are cut into looks exactly as the
live monitor does it (the whole picture when one person is in view, a crop
around the person when there are several; mirrored copies added), and run
through the same model, so what is learned is what the monitor will see.

Live, a trained activity is shown when 3 of the last 4 looks at a person
reach its threshold (about 2 s of evidence: on the example, 2 of 3 let
head-down writing through), is held for 4 s after the last supporting
look, and outranks the base model's label; a fall outranks it. The rule is
`min_hits` / `looks` under `custom_activities:` in
`config/model_config.yaml`, and training measures accuracy under the same
rule.
Files: `scripts/train_activities.py` (training),
`src/action_recognition/custom_activities.py` (the layer, live voting),
`src/action_recognition/clip_encoder.py` (the activity model).

## Limits

- It learns what is **visible in 16 frames (about half a second)**. An
  activity defined by duration ("asleep for 10 minutes") is the trained
  activity plus a rule on the logged duration, which the activity log and
  the API already give you.
- It recognises what it was **shown**. A camera angle, uniform or room
  very different from every training clip can be missed: add clips from it.
- Each activity is judged per person, so it needs people to be detected:
  heavily occluded or tiny people are not looked at.
