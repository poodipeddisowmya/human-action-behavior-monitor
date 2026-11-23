# Client demo (about 10 minutes)

Everything is shown from the dashboard, starting on its **Overview** page.
Written for the NVIDIA Jetson; on a laptop or a GPU server only the start
command differs (see the [README](../README.md#2-install-and-start)). What
the system is and how it works, in plain words:
[TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md).

## The day before

1. **Get the new-activity clips** (once per machine):
   `python scripts/fetch_new_activity_clips.py`
2. **Clean, steady power** (Jetson): the official adapter, and
   `sudo nvpmodel -m 1` (25 W); at maximum power the board can throttle.
3. **Decide who will be named.** Delete any test people in *Known people*.
   The demo shows naming someone live (step 3 below), so nobody needs to be
   enrolled in advance.

## 30 minutes before

1. **Start with an empty log:** stop the monitor if it runs, then
   `python scripts/reset_demo_log.py` (backs up the old log; enrolled people
   are kept).
2. **Free the Jetson's memory** (8 GB for everything): close browsers and
   desktop applications on the Jetson itself. With them open, the
   tea/tablets model can fail to load.
3. **Start it**, on recorded clips:
   ```bash
   PROBE_CAMERAS=0 ./run_jetson.sh --source data/samples/new-taking-tablets-1.mp4
   ```
   For a live USB camera instead: `./run_jetson.sh --source 0`.
4. **Open the dashboard from a laptop** on the same network:
   `http://<jetson-address>:8000` (`hostname -I` on the Jetson shows the
   address). Not through remote desktop: it re-encodes the screen and the
   video stutters.
5. **Fill the log:** on the Overview, click *Making tea*, *Taking tablets*
   and *Falling down* in turn, about 30 s each, then *Taking tablets* again.
   Every card now has real photos.

## The demo

### Opening (1 minute) — Overview

> "Cameras record everything, but nobody has time to watch them. This system
> watches for you: it finds every person, follows them, recognises what they
> are doing and who they are, and keeps photo proof of every activity. It is
> running here on this small NVIDIA Jetson, on site, so no video leaves the
> building."

Point to the four numbers at the top, then walk the four cards in order.

### 1. Same video, three machines (2 minutes)

Click **Open the comparison**, pick a clip (Cooking, Eating, Making tea).

> "The same clips through the same software on three machines. On a laptop
> CPU it recognises 1 of 29 clips; on a cloud Tesla T4 GPU 21; on this
> Jetson 22. The GPU is what makes it accurate, not just fast: the activity
> model looks several times a second instead of once every 1.5 seconds. And
> the Jetson matches the cloud GPU while staying on site."

### 2. New activities, no retraining (2 minutes)

Back on Overview, click **Making tea**, then **Taking tablets**, then
**Falling down**.

> "These three were not in the model. Making tea and taking tablets were
> added by describing them in words to a video transformer: no training
> videos, one line of configuration each. A fall is read from the skeleton:
> upright, then on the ground within a second. It shows as a red alert."

Point to the subtitle and the box label as each one appears.

### 3. Face recognition (3 minutes)

From the Source menu choose **Scenes → Classroom 4 phone**.

> "Everyone here is Unknown: the system only names people it has been taught,
> with their consent."

In **Faces on screen**, type a name under one face and press **Name**.

> "One click, and she is recognised: on the video, in the subtitle, and in
> the log, including what she did before I named her. From now on she is
> recognised on any camera. Nobody else is named, and the faces of people I
> do not name are never stored."

Optional: **Known people** → delete her: "and on request, everything about
her face is deleted at once."

### 4. Every activity logged, with proof (2 minutes)

Scroll to the **Activity log**; click a photo.

> "Every activity is logged: who, what, when, for how long, how sure, and
> photos of its start, middle and end. Filters show the new activities, the
> people we know, or the alerts. Everything here is also available through an
> API, to connect alarms or reports."

Click **Alerts** to show the fall.

### Close (30 seconds)

> "Next step: a few cameras on your site. We measure accuracy on your own
> footage, and fine-tune on it, which is the largest improvement available."

## Questions clients ask

| Question | Answer |
|---|---|
| How accurate is it? | On a GPU, 21–22 of 29 test clips clearly recognised; wrong labels under 1% of the time. On your own cameras we measure it on site. |
| Does video leave our building? | Not with the Jetson: everything runs on the device. |
| Can it learn our activities? | Ones the base model knows: one line. Others: describe them in words (as with tea and tablets), or fine-tune on a few hundred of your clips. |
| What about privacy? | Only people who agree are enrolled; others stay Unknown and their faces are never stored; enrolled people can be deleted at once. Proof photos are deleted after 30 days by default. |
| How many cameras? | Each running monitor processes one video stream (about 22 fps on this Jetson). More cameras: one device per camera, or a larger GPU server running several monitors. |
| Can it alert us? | Falls already show as alerts on the dashboard. Every event is also pushed live over the API, so an alarm, SMS or e-mail system can be connected with a small integration. |

## If something goes wrong

| Symptom | Fix |
|---|---|
| Making tea / taking tablets never appear | The log shows `[open-vocab] … failed`: free memory (step 2 above), restart. |
| Play buttons on the Overview are grey | The clips are missing: `python scripts/fetch_new_activity_clips.py`. |
| Names never appear | Faces too small or turned away; use the Classroom 4 phone clip. |
| Video stutters | You are on remote desktop: use the laptop's browser. |
| The board restarts | Power: the official adapter and 25 W mode. |
| The page does not open | Models take up to a minute to load after a start; then check the address. |
