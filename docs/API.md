# API Contract

Base URL: `http://localhost:8000`

## REST

### `GET /api/health`
```json
{ "status": "ok", "device": "cpu", "fps_target": 15 }
```

### `GET /api/persons`
List of everyone the system has seen, newest activity first.
```json
[
  { "id": "P001", "first_seen": "2026-08-30T10:30:10", "last_seen": "2026-08-30T10:32:20" }
]
```

### `GET /api/persons/{id}`
Full activity history for one person (last 200 events).
```json
{
  "id": "P001",
  "first_seen": "2026-08-30T10:30:10",
  "last_seen": "2026-08-30T10:32:20",
  "history": [
    { "label": "standing", "start_time": "...", "end_time": "...",
      "duration_s": 32.0, "confidence": 0.94 }
  ]
}
```

### `GET /api/events?limit=50&label=falling`
Global recent events, optionally filtered.

### `GET /api/stream/video`
MJPEG stream of the annotated video (bind directly to `<img src=...>`).

## WebSocket

### `WS /ws`
Server-push. Two message types:

```json
{ "type": "frame_annotations",
  "annotations": [{ "xyxy": [x1,y1,x2,y2], "person_id": 1,
                    "label": "walking", "confidence": 0.91 }] }
```

```json
{ "type": "event", "person_id": "P001", "label": "walking",
  "start_time": "...", "end_time": "...",
  "duration_s": 12.4, "confidence": 0.93 }
```

Client should send periodic pings to keep the socket alive; contents are ignored.
