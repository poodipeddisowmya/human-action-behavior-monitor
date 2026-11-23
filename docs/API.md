# API

Base URL: `http://<host>:8000`. Interactive documentation of every endpoint,
with request and response schemas, is served by the app itself at
**`/docs`**.

When `ACCESS_TOKEN` is set, every endpoint except `/api/health` needs it:
`?token=…` once in the browser (it sets a cookie), or the header
`Authorization: Bearer …`.

## Health and monitoring

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | The process answers (no token needed: for load balancers). |
| GET | `/api/ready` | 200 when models are loaded and frames are flowing, else 503. |
| GET | `/api/metrics` | FPS, detection time, people tracked, GPU memory, device. |

## Activity log

| Method | Path | Returns |
|---|---|---|
| GET | `/api/events?limit=50&label=falling%20down` | Newest logged activities: person, name, activity, start, duration, confidence, proof photo URLs. |
| GET | `/api/persons` | Everyone tracked, newest first. |
| GET | `/api/persons/{id}` | One person's activity history (`P001`). |
| GET | `/api/evidence/{event_id}/{index}` | A proof photo (JPEG; 0 = start, 1 = middle, 2 = end). |

## Video

| Method | Path | Returns |
|---|---|---|
| GET | `/api/stream/video` | The annotated video, MJPEG (use as an `<img src>`). |
| GET | `/api/stream/snapshot` | The current annotated frame, one JPEG. |
| GET | `/api/stream/live` | The labels on screen right now, per person. |
| GET | `/api/sources` | Clips, cameras, and what is playing. |
| POST | `/api/sources` | Switch source: `{"filename": "demo-cooking.mp4"}` or `"camera:0"`. |
| POST | `/api/sources/upload` | Upload a video (multipart `file`; mp4/avi/mov/mkv/webm, ≤ 500 MB). |

## Face recognition

| Method | Path | Does |
|---|---|---|
| GET | `/api/known-people` | Enrolled people and their face photos. |
| POST | `/api/known-people` | Add a person: `{"name": "Asha"}`. |
| POST | `/api/known-people/{id}/photos` | Enrol photos (multipart `files`); returns what was added and why others were refused. |
| POST | `/api/known-people/{id}/from-evidence` | Enrol the face in a proof photo: `{"event_id": 12, "index": 1}`. |
| GET | `/api/known-people/on-screen` | People in view with a clear face and no name. |
| GET | `/api/known-people/on-screen/{track}/face` | That person's face (JPEG). |
| POST | `/api/known-people/{id}/from-video` | Enrol someone in view: `{"track": 6}`; names them at once. |
| DELETE | `/api/known-people/{id}` | Delete a person and all their face data. |
| DELETE | `/api/known-people/{id}/photos/{photo_id}` | Delete one face photo. |

## Recorded comparison

| Method | Path | Returns |
|---|---|---|
| GET | `/api/showcase` | The CPU / T4 / Jetson figures and clip list. |
| GET | `/api/showcase/video/{side}/{file}` | One recorded video. |

## WebSocket

`WS /ws` pushes two kinds of message:

```json
{"type": "frame_annotations", "annotations": [{"person_id": 3, "name": "Asha",
  "label": "making tea", "confidence": 0.82, "xyxy": [x1, y1, x2, y2],
  "detail": null, "posture": "standing"}],
 "caption": {"text": "Asha is making tea in a kitchen.", "scene": "in a kitchen"}}

{"type": "event", "id": 412, "person_id": "P003", "identity": "Asha",
 "label": "making tea", "start_time": "2026-11-20T08:14:03Z",
 "end_time": "2026-11-20T08:14:21Z", "duration_s": 18.2, "confidence": 0.84,
 "evidence": ["/api/evidence/412/0", "/api/evidence/412/1", "/api/evidence/412/2"]}
```

`WS /ws/camera` receives the viewer's own webcam as JPEG frames ("Go live"
in the dashboard).
