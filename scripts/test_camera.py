"""A stand-in network camera: plays a video file as an IP camera stream.

For testing the CCTV path (connection, live database, reconnects) before the
real camera is installed, or on a machine without one. Serves the video,
looping at its own frame rate, as MJPEG over HTTP: the same kind of stream
many IP cameras offer next to RTSP, read by the monitor in exactly the same
way.

    python scripts/test_camera.py data/samples/office-people.mp4
    # then, in .env:   CAMERA_TEST=http://127.0.0.1:8554/video.mjpg
    # and start the monitor with  --source Test

Ctrl+C to stop. Stopping and starting it again shows the monitor's "no
signal" card and its automatic reconnection.
"""
from __future__ import annotations

import argparse
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2


class _Player:
    """Decodes the file once, at real-time speed, for every viewer."""

    def __init__(self, path: Path, width: int):
        self.path, self.width = path, width
        self.cond = threading.Condition()
        self.jpeg: bytes | None = None
        self.seq = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        cap = cv2.VideoCapture(str(self.path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        next_at = time.monotonic()
        while True:
            ok, img = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            h, w = img.shape[:2]
            if self.width and w > self.width:
                img = cv2.resize(img, (self.width, round(h * self.width / w)), interpolation=cv2.INTER_AREA)
            data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
            with self.cond:
                self.jpeg, self.seq = data, self.seq + 1
                self.cond.notify_all()
            next_at += 1 / fps
            time.sleep(max(0.0, next_at - time.monotonic()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("--port", type=int, default=8554)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from other machines")
    ap.add_argument("--width", type=int, default=1280, help="scale wider video down to this")
    args = ap.parse_args()
    if not args.video.is_file():
        raise SystemExit(f"No such video: {args.video}")
    player = _Player(args.video, args.width)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.split("?")[0] != "/video.mjpg":
                self.send_error(404, "the stream is at /video.mjpg")
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seen = 0
            try:
                while True:
                    with player.cond:
                        player.cond.wait_for(lambda: player.seq != seen, timeout=5)
                        seen, data = player.seq, player.jpeg
                    if data is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     + f"Content-Length: {len(data)}\r\n\r\n".encode() + data + b"\r\n")
            except OSError:
                pass                                   # viewer left

    # Stopped by a script (kill): leave at once. The default handling let
    # the decoder thread die mid-frame, and OpenCV aborted with a core dump.
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"Test camera playing {args.video.name} at http://{args.host}:{args.port}/video.mjpg  (Ctrl+C stops)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
