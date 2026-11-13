"""Post-deploy smoke test: is a running monitor working, and is it real time?

    python scripts/smoke_test.py --base http://127.0.0.1:8000 --token XYZ
    python scripts/smoke_test.py --base https://abc.trycloudflare.com --token XYZ --json out.json

Checks, against a live server (locally or through a tunnel):
    auth          no token -> 401, token -> 200, health open
    ready         /api/ready answers 200
    realtime      playback FPS and detection latency from /api/metrics
    stream        frames/s actually delivered over the MJPEG stream
    webcam        browser-camera path: frames pushed over /ws/camera, ack
                  round-trip, and a label coming back for the pushed person
    multi-person  FPS and latency on a crowded clip
Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import httpx
import websockets

ROOT = Path(__file__).resolve().parent.parent

# Real time = the video plays at (nearly) its own native frame rate, and
# boxes refresh several times a second: detection under 150 ms keeps them
# on the person. Clips run from 10 fps (surveillance) to 60 fps, so the bar
# is relative to each clip, capped at 25 fps.
REALTIME_RATIO = 0.85
MAX_DETECT_MS = 150.0
MIN_STREAM_FPS = 8.0         # through a public tunnel; ~20 locally


def realtime(m: dict) -> bool:
    native = min(float(m.get("source_fps") or 25), 25.0)
    return m.get("playback_fps", 0) >= REALTIME_RATIO * native


class Checks:
    def __init__(self):
        self.results: list[dict] = []

    def add(self, name: str, ok: bool, **detail) -> None:
        self.results.append({"check": name, "ok": bool(ok), **detail})
        flag = "PASS" if ok else "FAIL"
        extra = "  ".join(f"{k}={v}" for k, v in detail.items())
        print(f"  [{flag}] {name:<14} {extra}", flush=True)

    @property
    def ok(self) -> bool:
        return all(r["ok"] for r in self.results)


def metrics(c: httpx.Client) -> dict:
    return c.get("/api/metrics").json()


def wait_fresh_metrics(c: httpx.Client, seconds: float = 12) -> dict:
    """Metrics roll over every 5 s; wait long enough for a full window."""
    time.sleep(seconds)
    return metrics(c)


def stream_fps(base: str, headers: dict, seconds: float = 8.0) -> tuple[float, int]:
    frames, size = 0, 0
    t0 = time.time()
    with httpx.stream("GET", f"{base}/api/stream/video", headers=headers, timeout=20) as s:
        for chunk in s.iter_bytes():
            frames += chunk.count(b"--frame")
            size += len(chunk)
            if time.time() - t0 >= seconds:
                break
    dt = time.time() - t0
    return frames / dt, int(size / dt / 1024)


async def webcam(base: str, headers: dict, clip: Path, seconds: float = 15.0) -> dict:
    ws_url = base.replace("https://", "wss://").replace("http://", "ws://") + "/ws/camera"
    cap = cv2.VideoCapture(str(clip))
    rtts: list[float] = []
    async with websockets.connect(ws_url, extra_headers=headers, max_size=None) as ws:
        httpx.post(f"{base}/api/sources", headers=headers,
                   json={"filename": "browser:camera"}, timeout=10)
        t0 = time.time()
        while time.time() - t0 < seconds:
            ok, f = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            scale = min(1.0, 960 / f.shape[1])
            if scale < 1:
                f = cv2.resize(f, (int(f.shape[1] * scale), int(f.shape[0] * scale)))
            ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 72])
            ts = time.time()
            await ws.send(jpg.tobytes())
            await ws.recv()
            rtts.append((time.time() - ts) * 1000)
            await asyncio.sleep(max(0, 1 / 15 - (time.time() - ts)))
        live = httpx.get(f"{base}/api/stream/live", headers=headers, timeout=10).json()
    dt = time.time() - t0
    return {"sent_fps": round(len(rtts) / dt, 1),
            "ack_ms_p50": round(statistics.median(rtts)) if rtts else None,
            "ack_ms_p90": round(sorted(rtts)[int(len(rtts) * 0.9)]) if rtts else None,
            "people": live.get("people", [])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--token", default="")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--min-stream-fps", type=float, default=MIN_STREAM_FPS)
    args = ap.parse_args()
    base = args.base.rstrip("/")
    H = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    samples = ROOT / "data" / "samples"
    chk = Checks()
    print(f"\nSmoke test against {base}\n", flush=True)

    with httpx.Client(base_url=base, headers=H, timeout=15) as c:
        h = httpx.get(f"{base}/api/health", timeout=15)
        chk.add("health", h.status_code == 200, device=h.json().get("device"),
                gpu=h.json().get("gpu"))
        if args.token:
            r = httpx.get(f"{base}/api/sources", timeout=15)
            chk.add("auth", r.status_code == 401 and c.get("/api/sources").status_code == 200,
                    without_token=r.status_code)

        t0 = time.time()
        while time.time() - t0 < 120 and c.get("/api/ready").status_code != 200:
            time.sleep(2)
        chk.add("ready", c.get("/api/ready").status_code == 200)

        # Single-person activity clip.
        c.post("/api/sources", json={"filename": "demo-cooking.mp4"})
        m = wait_fresh_metrics(c)
        chk.add("realtime", realtime(m) and 0 < m.get("detect_ms", 1e9) <= MAX_DETECT_MS,
                playback_fps=m.get("playback_fps"), source_fps=m.get("source_fps"),
                detect_ms=m.get("detect_ms"),
                device=m.get("gpu") or m.get("device"), gpu_mem_mb=m.get("gpu_mem_mb"))
        live = c.get("/api/stream/live").json()
        chk.add("labels", any(p["label"] not in ("detecting…", "-") for p in live["people"]),
                people=[f"{p['person_id']}:{p['label']}" for p in live["people"]],
                caption=live.get("caption"))

        fps, kbps = stream_fps(base, H)
        chk.add("stream", fps >= args.min_stream_fps, fps=round(fps, 1), kB_per_s=kbps)

        # Browser webcam path, using a clip as the "camera".
        try:
            w = asyncio.run(webcam(base, H, samples / "demo-using-computer.avi"))
            m = metrics(c)
            chk.add("webcam", w["sent_fps"] >= 8 and bool(w["people"]), **w,
                    detect_ms=m.get("detect_ms"))
        except Exception as e:
            chk.add("webcam", False, error=repr(e))

        # Crowded scenes: the per-person path, several tracks at once.
        for crowd in ("office-people.mp4", "store-aisle.mp4"):
            if not (samples / crowd).exists():
                continue
            c.post("/api/sources", json={"filename": crowd})
            m = wait_fresh_metrics(c)
            chk.add("multi-person", realtime(m) and m.get("detect_ms", 1e9) <= MAX_DETECT_MS * 2,
                    clip=crowd, playback_fps=m.get("playback_fps"),
                    source_fps=m.get("source_fps"), detect_ms=m.get("detect_ms"),
                    tracked=m.get("tracked"))
        m = metrics(c)
        chk.add("stability", m.get("restarts", 0) == 0 and m.get("pipeline_alive"),
                restarts=m.get("restarts"), detect_errors=m.get("detect_errors"))
        c.post("/api/sources", json={"filename": "demo-cooking.mp4"})

    print(f"\n  {'ALL PASS' if chk.ok else 'SOME CHECKS FAILED'}\n", flush=True)
    if args.json:
        args.json.write_text(json.dumps({"base": base, "ok": chk.ok,
                                         "checks": chk.results}, indent=2, default=str))
    return 0 if chk.ok else 1


if __name__ == "__main__":
    sys.exit(main())
