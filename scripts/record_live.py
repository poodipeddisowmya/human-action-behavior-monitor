"""Record what the dashboard shows: the running monitor's annotated video.

Saves the live stream (boxes, names, activities and the subtitle, exactly as
on screen) to MP4, from whatever is playing: a CCTV camera, a USB camera or a
clip. Optionally switches the monitor to each clip first, so a set of
recordings is made in one go.

    python scripts/record_live.py --seconds 30 --out recordings/       # what is playing now
    python scripts/record_live.py --seconds 15 --out recordings/ \\
        --switch office-desk-work.mp4 office-phone-call.mp4             # each clip in turn

Add --token XYZ when the monitor has ACCESS_TOKEN set.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# Frames from right after a switch still show the previous source.
_SETTLE_S = 1.5


def _open(url: str, token: str | None, data: bytes | None = None):
    req = urllib.request.Request(url, data, {"content-type": "application/json"} if data else {})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req, timeout=15)


def record(base: str, token: str | None, seconds: float, out: Path, fps: float) -> int:
    """Write `seconds` of the MJPEG stream to `out` at a steady `fps`."""
    stream = _open(f"{base}/api/stream/video", token)
    buf = b""
    writer = None
    frame = None
    written = 0
    t0 = time.monotonic()
    next_at = t0
    while time.monotonic() - t0 < seconds:
        buf += stream.read(65536)
        # Keep only the newest complete JPEG in the buffer.
        end = buf.rfind(b"\xff\xd9")
        start = buf.rfind(b"\xff\xd8", 0, end) if end > 0 else -1
        if start >= 0:
            img = cv2.imdecode(np.frombuffer(buf[start:end + 2], np.uint8), cv2.IMREAD_COLOR)
            buf = buf[end + 2:]
            if img is not None:
                frame = img
        if frame is None:
            continue
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        # A steady frame rate whatever the stream's: repeat or skip frames.
        while time.monotonic() >= next_at:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
            written += 1
            next_at += 1 / fps
    stream.close()
    if writer is not None:
        writer.release()
    return written


def _ffmpeg() -> str | None:
    """ffmpeg on the PATH, else the one bundled with imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def to_h264(path: Path) -> bool:
    """Re-encode to H.264 in place, so the file plays in every browser
    (OpenCV can only write MPEG-4 Part 2, which browsers do not play)."""
    ffmpeg = _ffmpeg()
    if not ffmpeg:
        return False
    tmp = path.with_suffix(".h264.mp4")
    r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264",
                        "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", "-an", str(tmp)])
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(path)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--token", default=None)
    ap.add_argument("--seconds", type=float, default=20.0, help="per recording")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--out", type=Path, default=Path("recordings"))
    ap.add_argument("--name", default=None, help="file name (without .mp4) when not switching")
    ap.add_argument("--switch", nargs="*", default=None,
                    help="clips in data/samples (or 'stream:<camera>') to record in turn")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for source in args.switch or [None]:
        if source:
            _open(f"{args.base}/api/sources", args.token,
                  json.dumps({"filename": source}).encode()).read()
            time.sleep(_SETTLE_S)
            name = Path(source.replace("stream:", "camera-")).stem
        else:
            name = args.name or time.strftime("live-%Y%m%d-%H%M%S")
        out = args.out / f"{name}.mp4"
        n = record(args.base, args.token, args.seconds, out, args.fps)
        codec = "H.264" if to_h264(out) else "MPEG-4 (no ffmpeg: may not play in a browser)"
        print(f"{out}  {n} frames, {n / args.fps:.0f} s, {codec}")


if __name__ == "__main__":
    main()
