# CCTV and IP cameras

The monitor reads a CCTV or IP camera directly over the network (RTSP or
HTTP), on the Jetson or any other machine running it. No recorder (NVR) or
cloud service sits in between, and the video stays on site.

## 1. Find the camera's stream address

Every IP camera has an RTSP address. Use the **sub-stream** (720p or D1, 15–25
fps): the models work at 640 px, so a 4K main stream only costs decoding time.

| Make | Sub-stream address (port 554) |
|---|---|
| Hikvision | `rtsp://user:pass@<ip>:554/Streaming/Channels/102` (main: `101`) |
| Dahua / CP Plus | `rtsp://user:pass@<ip>:554/cam/realmonitor?channel=1&subtype=1` |
| Uniview | `rtsp://user:pass@<ip>:554/unicast/c1/s1/live` |
| Axis | `rtsp://user:pass@<ip>/axis-media/media.amp?resolution=1280x720` |
| Other | the camera's manual, or its web page under *Network → RTSP* |

Create a **view-only user** on the camera for the monitor; do not use the
admin account. If the password contains `@ : / # ?`, write it URL-encoded
(`@` → `%40`, `#` → `%23`).

Check the address from the Jetson before going further:

```bash
.venv/bin/python -c "import cv2,sys; c=cv2.VideoCapture(sys.argv[1]); print('OK', c.read()[1].shape) if c.isOpened() else print('cannot open')" "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102"
```

## 2. Name it in `.env`

One line per camera, `CAMERA_<NAME>=<address>`:

```
CAMERA_RECEPTION=rtsp://viewer:secret@192.168.1.64:554/Streaming/Channels/102
CAMERA_PANTRY=rtsp://viewer:secret@192.168.1.65:554/cam/realmonitor?channel=1&subtype=1
```

`CAMERA_FRONT_DESK` appears as *Front Desk*. The address and password stay on
this machine: the dashboard, the API, the database and the log only show
the name and the camera's IP address.

## 3. Start on it

```bash
./run.sh --source Reception                 # Linux, any board
run.bat --source Reception                  # Windows
```

Or start as usual and pick it in the dashboard's **Source** menu, under
*Live cameras* (marked *CCTV*). The video panel shows the camera's name, its
IP address and **LIVE**.

## What happens when

| Situation | What the monitor does |
|---|---|
| Normal | Reads the camera continuously and processes the newest frame, so the picture is never more than a frame behind, however busy the models are. |
| Network blip, camera reboot, cable pulled | The panel shows **NO SIGNAL · RECONNECTING** and a "No signal" picture (not a frozen last frame). It reconnects by itself, retrying after 1, 2, 5, then every 10 s. Activities in progress are closed and saved. |
| Camera comes back | Back to **LIVE** within a few seconds; nothing to restart. |
| Wrong address or password | Same as no signal; the log (`logs/monitor.log`) says `camera Reception: cannot connect` with the password hidden. |

`GET /api/metrics` → `camera` reports `live`, `connecting` or `no signal`,
for how long, and the number of reconnections: point a monitoring system at
it.

## Where the results go

Every activity is saved the moment it ends, with the camera's name, to the
database (`logs/monitor.db`, table `actions`) and its proof photos to
`data/evidence/`. The dashboard's **Database** tab shows the rows arriving
live. `GET /api/events` and the WebSocket `/ws` give the same data to other
systems ([API.md](API.md)).

## Several cameras

One running monitor processes one camera at a time; the Source menu switches
between them. To watch several at once, run one monitor per camera, each with
its own `API_PORT` and `--source` (a Jetson Orin Nano runs one at about 22
fps; a desktop GPU runs several).

## Rehearsing without a camera

`scripts/test_camera.py` plays a video file as a network camera (MJPEG over
HTTP, read exactly like a real one):

```bash
python scripts/test_camera.py data/samples/office-people.mp4
# .env:  CAMERA_TEST=http://127.0.0.1:8554/video.mjpg
./run.sh --source Test
```

Stop and restart it to see the "no signal" state and the reconnection.

## Placement, for the best results

- **Height 2.5–3 m, angled down 15–30°.** Higher and steeper, people are
  seen from above and activities are harder to tell apart.
- **People at least 1/4 of the picture's height** for activities, and faces
  at least 40 px wide for face recognition: a camera over a reception desk
  or a pantry counter, rather than one covering a whole floor.
- **Even light, no strong backlight** (a window behind people turns them into
  silhouettes).
- **Fixed, not pan-tilt:** a moving camera breaks tracking.
