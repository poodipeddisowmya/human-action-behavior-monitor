# Indian office demo recordings

What the dashboard showed on the NVIDIA Jetson Orin Nano Super, recorded on
2026-11-23: four clips of an Indian corporate office, each played into the
monitor **as a network camera** (`scripts/test_camera.py`, read by the same
code as an RTSP CCTV camera, see [docs/CCTV.md](../../docs/CCTV.md)), and
the annotated video saved with `scripts/record_live.py`. Shown on the
dashboard's Overview, card 5. Descriptions: `recordings.json`.

These are stock clips, not footage from a CCTV camera: they are filmed at
eye level, closer than a ceiling camera would be.

## How the six were chosen

19 clips of the same office shoot were measured on the Jetson, sampling the
labels on screen for 20 s each. Four kinds of mistake showed up and were
fixed in the software (not by choosing clips):

| Mistake | Fix |
|---|---|
| A desk-phone call read as *smoking* (86%) | X-CLIP's look at the person settles it: now *using phone* (84%) |
| Writing on paper at a desk read as *using computer* | The same check: *writing* now shows (44–48%) |
| Colleagues chatting over mugs read as *making tea* (71%) | "Talking" added to X-CLIP's other descriptions: 37% |
| Seated people read as *running* when swivelling | *Running* now needs an upright skeleton |

The four recorded here are those whose labels and subtitles were then
correct throughout. Not recorded, because they are still partly wrong:

| Clip | What goes wrong |
|---|---|
| A team at their desks, one walks over | the walking man is labelled *sitting* for a while |
| Going through papers | alternates *using computer* / *writing*; the subtitle lags the label |
| Colleagues with mugs | *eating* 20% of the time |
| A woman with a mug on a call | *making tea* |
| A pen held to the mouth | *smoking* |

## Credits

Adaptations (boxes, labels and subtitles drawn on) of videos by the **RDNE
Stock project** on Pexels, used under the
[Pexels License](https://www.pexels.com/license/):

| Recording | Original |
|---|---|
| office-floor | https://www.pexels.com/video/people-working-on-office-using-computer-7581202/ |
| office-desk-work | https://www.pexels.com/video/man-analyzing-documents-7581201/ |
| office-computer-work | https://www.pexels.com/video/man-busy-with-paperworks-7581303/ |
| office-phone-call | https://www.pexels.com/video/man-talking-on-the-telephone-7581273/ |

The original clips themselves are not in the repository;
`python scripts/fetch_new_activity_clips.py` downloads them into
`data/samples/office-*.mp4`.
