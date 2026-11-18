# New activities: making tea, taking tablets, falling down

The activity model (VideoMAE, trained on Kinetics-400) only knows the
activities it was trained on. Three more were needed:

| Activity | Recognised by | Why that one |
|---|---|---|
| **Making tea** | X-CLIP, from a text description | Kinetics folds it into cooking |
| **Taking tablets** | X-CLIP, from a text description | No video dataset the models know has it |
| **Falling down** | The skeleton over time (`fall_detector.py`) | A fall lasts about a second, too short for a look; its end state is unmistakable |

All three show in the live view, the subtitles ("A person is taking tablets
in a living room.", "Alert: a person has fallen down in an office."), the
events log with proof images, and the comparison tab.

## Open-vocabulary activities (X-CLIP)

[X-CLIP](https://huggingface.co/microsoft/xclip-base-patch16-zero-shot) is a
video transformer trained to match video with text. Each look (32 frames,
2.5 s) is scored against the configured activities **and** against the
activities the system already tracks (eating, drinking, cooking...). The
softmax over all of them is what stops someone drinking water from scoring as
taking tablets: the drinking description wins that look.

A label shows when 2 of the last 3 looks reach the activity's bar, and holds
4 s after the last supporting look. One look a second, in its own thread:
the video and the other models never wait for it. GPU only.

**Adding an activity is one entry** in `config/model_config.yaml`:

```yaml
open_vocab:
  activities:
    watering plants:
      - watering plants
      - a person watering plants with a watering can
```

Then measure it before trusting it: put a few real clips of it in
`data/new-activities/`, list them in `scripts/eval_open_vocab.py`, and run

```bash
python scripts/eval_open_vocab.py --model microsoft/xclip-base-patch16-zero-shot --fp16
```

It reports, per activity, how often it is found on clips that show it and
how often it wrongly wins on everything else (the 29 showcase clips, everyday
clips). Set its bar (`thresholds:`) in the gap between the two.

### Measured (Jetson Orin Nano, 2026-11-18)

Score of the activity, per look, on 47 clips:

| | On clips showing it | Highest on any other clip | False alarms |
|---|---|---|---|
| Taking tablets | 0.43–0.65 (90th pct) | 0.14 (a drinking clip) | 0 of 45 clips |
| Making tea | 0.63–0.87 | 0.23 (a cooking clip) | 0 of 44 clips |

The smaller `xclip-base-patch32` was tried first: 4x faster (78 ms a look)
but tablets scored 0.18–0.33 against 0.20 for a drinking clip, no margin.
The zero-shot model costs 313 ms and 582 MB a look on the Orin Nano (FP16).

Through the full pipeline on the Jetson (`scripts/record_showcase.py`, all
38 showcase clips, 2026-11-18):

| Clip | Activity | On screen | First shown |
|---|---|---|---|
| Elderly man takes pills with water | taking tablets | 64% | 4.6 s |
| Young woman takes pills with water | taking tablets | 21% | 8.8 s |
| Man making chai at a stall | making tea | 59% | 5.1 s |
| Tea brewed in an urn | making tea | 25% | 9.0 s |
| Moroccan tea made outdoors | making tea | 44% | 4.9 s |
| Man falls in the snow | falling down | 53% | 2.5 s |

None of the other 32 clips was labelled with a new activity at any moment.
The 29 labelled ones score as before (23/29 clearly recognised, 70% correct
on screen, 0.9% wrong; the T4 scores the same), but on the Orin Nano the
first label comes ~0.6 s later: X-CLIP shares its small GPU with the
activity model. See [JETSON.md](JETSON.md).

**Weakest: taking tablets on the young woman (21%).** She takes the pill in
the first two seconds, before X-CLIP has a full 2.5 s look; what follows is
mostly drinking. Measure on footage from your own site before relying on
it (`scripts/eval_open_vocab.py`).

### Running beside the other models on an 8 GB Jetson

X-CLIP needs ~600 MB of GPU memory. With a desktop session open the Jetson
ran out, and the TensorRT person detector failed first (40 "CUDA error: out
of memory" on one crowded clip). Three changes fixed it: X-CLIP releases its
working memory after each look (PyTorch would otherwise keep it reserved,
where TensorRT cannot use it); it loads last in the warm-up, after evicting
the page cache the other weights filled; and a look or a load that fails for
memory is retried. A production unit should still run headless.

## Falling down (skeleton)

A fall is a change of state, not a speed. Measured on 9 real falls and the
everyday clips around them:

| | Falls (9) | Everyday (4) |
|---|---|---|
| Box width ÷ height at the end | 1.26–2.40 | 0.56–0.86 |
| Upright → lying-wide | 0.13–0.60 s | never |
| Fastest drop of the box top (body-heights/s) | 0.46–1.39 | 0.38–0.64 |

Speed overlaps; the end state does not. The detector needs the person seen
upright, then lying within 1.5 s (box ≥ 1.2x wider than tall, and torso near
horizontal or height down to under 60% of standing), holding in most frames
for 0.4 s. It then shows until they have been up again for a second, and at
least 5 s. A new track ID where one just vanished inherits its history: a
falling body changes shape so fast the tracker often renumbers it.

**Result: 9 of 9 falls, 0 false alarms on 43 other clips** (4 everyday
clips from the same rooms, dancing crowds, push-ups, squats, swimming,
cooking close-ups, the new tea and tablet clips).

Each rule exists because a clip broke the one before it: a hand reaching
into a cooking close-up (box shape alone, no torso), a dancer with arms out
in a crowd (one noisy skeleton frame), a fall towards the camera (torso
looks steep; the height collapse shows it).

Not caught: a fall that ends sitting against something (the box stays
tall), or one hidden behind furniture. A quick, deliberate lie-down on a
sofa looks like a fall; for a safety alert that is the side to err on.

## Proof images in the events log

For every logged activity, up to three shots of the person: the start, the
middle and the end, each a crop with context, their box and a banner
(person, activity, confidence, time). One shot a second at most while the
activity lasts; the first, middle and last are kept.

- Saved as JPEGs (~30–60 KB) under `data/evidence/<date>/`, linked from the
  `actions` table (`evidence` column; older databases gain it on start-up).
- Served by `/api/evidence/<action id>/<n>`, behind `ACCESS_TOKEN` like the
  rest of the API; only files under `data/evidence/` can be reached.
- Shown as thumbnails in the events table and each person's history; click
  for the full-size viewer (arrow keys between shots, Esc to close).
- Deleted after `EVIDENCE_RETENTION_DAYS` (default 30; 0 keeps them).
  `EVIDENCE=0` turns it off.

These are images of people: set the retention to what your site's policy
allows, and `ACCESS_TOKEN` on any server others can reach.

## Test clips and licences

`data/new-activities/` (git-ignored; see its `SOURCES.md`): Mixkit (free
licence, raw files not to be re-shared) and Wikimedia Commons (CC BY / BY-SA)
clips for the comparison, and 12 clips from the UR Fall Detection Dataset
(CC BY-NC-SA 4.0, non-commercial research) used **only** to measure the
fall detector locally, never shown or shipped.
