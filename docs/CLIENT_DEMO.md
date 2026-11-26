# Client demo (about 13 minutes)

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
4. **Connect the CCTV camera** if one is available: its sub-stream address in
   `.env` as `CAMERA_RECEPTION=rtsp://…`, checked as in
   [CCTV.md](CCTV.md#1-find-the-cameras-stream-address). Point it at a desk
   or the pantry, 2.5–3 m high. Without one, step 5 uses the recorded office
   clips.

## 30 minutes before

1. **Start with an empty log:** stop the monitor if it runs, then
   `python scripts/reset_demo_log.py` (backs up the old log; enrolled people
   are kept).
2. **Free the Jetson's memory** (8 GB for everything): close browsers and
   desktop applications on the Jetson itself. With them open, the
   tea/tablets model can fail to load.
3. **Start it**, on recorded clips:
   ```bash
   PROBE_CAMERAS=0 ./run.sh --source data/samples/new-taking-tablets-1.mp4
   ```
   On the CCTV camera instead: `./run.sh --source Reception`; a USB
   camera: `./run.sh --source 0`.
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

### 5. Your CCTV camera, stored as it happens (3 minutes)

Overview, card 5: play **Phone call**, then **Office floor**.

> "An Indian office, fed in as a CCTV camera would be. Desk work, a phone
> call, a whole floor of people: each recognised live on this Jetson."

Click **Watch the database fill**. In the Source menu pick the CCTV camera
(or *Office → Floor* without one); have someone sit at the desk, make a
call, then stand up and walk off.

> "On the left, what the camera sees and what is going on now. The moment
> an activity ends, it is written to the database: here is the row, with
> the camera's name, who, how long, how sure, and the photos. The count of
> rows and the database size go up as we watch. Your systems read the same
> rows through the API."

Optional: unplug the camera's network cable for ten seconds. The video
shows **NO SIGNAL · RECONNECTING**, and it comes back by itself when the
cable goes back in.

### Close (30 seconds)

> "Next step: a few cameras on your site. We measure accuracy on your own
> footage, and train it on clips from your cameras, which is the largest
> improvement available."

## Questions clients ask

| Question | Answer |
|---|---|
| How accurate is it? | On a GPU, 21–22 of 29 test clips clearly recognised; wrong labels under 1% of the time. On your own cameras we measure it on site. |
| Does video leave our building? | Not with the Jetson: everything runs on the device. |
| Can it learn our activities? | Yes: put 20-50 clips of the activity from your cameras in a folder and run one command; it trains in minutes on this Jetson and reports its accuracy on clips it has not seen ([TRAINING.md](TRAINING.md)). Activities the base model knows: one line; describable ones: one entry. |
| What about privacy? | Only people who agree are enrolled; others stay Unknown and their faces are never stored; enrolled people can be deleted at once. Proof photos are deleted after 30 days by default. |
| Can it use our existing CCTV cameras? | Yes: any IP camera with an RTSP address (Hikvision, Dahua / CP Plus, Uniview, Axis…), one line each in `.env`. No new cameras or recorder needed; it reads the camera's second stream alongside your NVR. |
| Where is the data? | In a database on the device (`logs/monitor.db`), proof photos in `data/evidence/`. Nothing is sent anywhere unless you connect a system to the API. |
| How many cameras? | Each running monitor processes one video stream (about 22 fps on this Jetson). More cameras: one device per camera, or a larger GPU server running several monitors. |
| Can it alert us? | Falls already show as alerts on the dashboard. Every event is also pushed live over the API, so an alarm, SMS or e-mail system can be connected with a small integration. |

## If something goes wrong

| Symptom | Fix |
|---|---|
| Making tea / taking tablets never appear | The log shows `[open-vocab] … failed`: free memory (step 2 above), restart. |
| Play buttons on the Overview are grey | The clips are missing: `python scripts/fetch_new_activity_clips.py`. |
| Names never appear | Faces too small or turned away; use the Classroom 4 phone clip. |
| CCTV camera shows NO SIGNAL | Wrong address or password, or the camera is unreachable: test the address as in [CCTV.md](CCTV.md#1-find-the-cameras-stream-address); `logs/monitor.log` says why. |
| Video stutters | You are on remote desktop: use the laptop's browser. |
| The board restarts | Power: the official adapter and 25 W mode. |
| The page does not open | Models take up to a minute to load after a start; then check the address. |
