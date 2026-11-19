# Client demo on the Jetson (about 10 minutes)

## Before the client arrives

1. **Free the Jetson's memory.** It has 8 GB for everything. Close browsers,
   the Claude app and the software updater on the Jetson itself; watch the
   demo from a laptop instead (below). With them open, the tea/tablet model
   can fail to load (the log says `[open-vocab] load failed ... retrying`).
2. **Steady power.** `sudo nvpmodel -m 1` (25 W) on the official adapter;
   in MAXN_SUPER the board throttles and has restarted under load.
3. **Start it:**
   ```bash
   ./run_jetson.sh                                    # demo clips
   ./run_jetson.sh --source 0                         # or the USB camera
   ```
4. **Open the dashboard on a laptop on the same network:**
   `http://<jetson-ip>:8000` (`hostname -I` on the Jetson). Not through
   remote desktop: it re-encodes the screen and the video stutters.
5. **Enrol yourself** (Known people → Add person, 3 clear photos), so the
   live camera part names you.

## The demo

| # | Show | Where | Say |
|---|---|---|---|
| 1 | Live detection | Live monitor, a demo clip | Every person is boxed, tracked with an ID, labelled with an activity; the subtitle describes the scene. |
| 2 | Proof images | Recent events → click a thumbnail | Each logged activity keeps 3 shots of the person — start, middle, end — as evidence. |
| 3 | Face recognition | Live camera (`--source 0`), you in view | Enrolled people are named: "Asha is using a phone". Strangers stay Unknown. Enrol straight from a proof image: "This face is … → Enrol face". |
| 4 | New activities | CPU vs GPU tab → the NEW clips | Making tea and taking tablets are defined in words to a video transformer — no training. Falls come from the skeleton: 9 of 9 real falls, no false alarms. |
| 5 | Why a GPU | CPU vs GPU tab, any clip | Same clips, same software: laptop CPU 1/29 clearly recognised, Tesla T4 and Jetson 23/29, the three videos frame-synced. |
| 6 | Edge vs cloud | Same tab, Jetson panel | The Jetson matches the cloud T4 at 25 W, on site, with no video leaving the building. |

Good camera practice for the live part: 1.5–3 m away, upper body and hands
in view, light in front of the person, face visible.

## If something goes wrong

| Symptom | Fix |
|---|---|
| Tea / tablets never appear | Log shows `[open-vocab] ... failed`: free memory (step 1), restart. |
| Names never appear | Nobody enrolled, or faces too small/turned away: enrol 3 photos, move closer. |
| Video stutters | You are on remote desktop: use the laptop browser. |
| Board restarts | Power: step 2. |
