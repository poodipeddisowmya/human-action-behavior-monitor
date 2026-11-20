# Client demo on the Jetson (about 10 minutes)

## Before the client arrives

1. **Free the Jetson's memory.** It has 8 GB for everything. Close browsers,
   the Claude app and the software updater on the Jetson itself; watch the
   demo from a laptop instead (below). With them open, the tea/tablet model
   can fail to load (the log says `[open-vocab] load failed ... retrying`).
2. **Steady power.** `sudo nvpmodel -m 1` (25 W) on the official adapter;
   in MAXN_SUPER the board throttles and has restarted under load.
3. **Start it** (recorded clips; `PROBE_CAMERAS=0` hides the camera button):
   ```bash
   PROBE_CAMERAS=0 ./run_jetson.sh --source data/samples/new-taking-tablets-1.mp4
   ./run_jetson.sh --source 0                         # or the USB camera
   ```
   The new-activity clips are local copies in `data/samples/new-*`
   (git-ignored: the stock footage may not be re-shared). Without them the
   Overview's play buttons are greyed out.
4. **Open the dashboard on a laptop on the same network:**
   `http://<jetson-ip>:8000` (`hostname -I` on the Jetson). Not through
   remote desktop: it re-encodes the screen and the video stutters.
5. **Enrol yourself** (Known people → Add person, 3 clear photos), so the
   live camera part names you.

## The demo

The dashboard opens on **Overview**: four numbered cards, each with real
results from this machine and a button that shows it working. Walk them in
order.

| # | Card | Click | Say |
|---|---|---|---|
| 1 | Same video, three machines | Open the comparison | Same 29 clips, same software: laptop CPU 1/29 clearly recognised, Tesla T4 21/29, Jetson 22/29, the three videos frame-synced. The Jetson matches the cloud GPU, on site, with no video leaving the building. |
| 2 | New activities, no retraining | Making tea / Taking tablets / Falling down | Tea and tablets are defined in words to a video transformer, no training clips. Falls come from the skeleton and show as an alert in the subtitle and the log. |
| 3 | Face recognition | See it live, then Name people on screen | Enrolled people are named on the box, the subtitle, the log and the photos; everyone else stays Unknown. Name anyone in view: switch to a clip with several people (Classroom 4 phone), type a name under their face in Faces on screen, and the box, subtitle and log name them at once. With consent only. |
| 4 | Every activity logged, with proof photos | Click a photo, then Open the activity log | Each activity keeps photos of the start, middle and end. The log filters to new activities, known people or alerts. |

A fresh log for the day: stop the server, back up `logs/monitor.db`, and
clear its `actions` and `persons` tables (enrolled people are kept), then
play each new-activity clip once so every card has photos.

Good camera practice for the live part: 1.5–3 m away, upper body and hands
in view, light in front of the person, face visible.

## If something goes wrong

| Symptom | Fix |
|---|---|
| Tea / tablets never appear | Log shows `[open-vocab] ... failed`: free memory (step 1), restart. |
| Names never appear | Nobody enrolled, or faces too small/turned away: enrol 3 photos, move closer. |
| Video stutters | You are on remote desktop: use the laptop browser. |
| Board restarts | Power: step 2. |
