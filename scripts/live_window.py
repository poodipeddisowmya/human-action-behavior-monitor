"""Watch the running monitor in a plain desktop window: boxes, IDs, labels
and the subtitle, as the dashboard shows them.

    python scripts/live_window.py                     # this machine's server
    python scripts/live_window.py --base http://jetson:8000 --fullscreen

Reads the server's MJPEG stream (/api/stream/video), whose frames already have
everything drawn in. Uses ~100 MB, against ~600 MB for a browser: on an 8 GB
Jetson that memory belongs to the models. Keys: f full screen, q / Esc quit.
"""
from __future__ import annotations

import argparse
import http.client
import time
import urllib.request

import cv2
import numpy as np

WINDOW = "Human Action Monitor - live"


def frames(url: str):
    """JPEG frames from a multipart MJPEG stream, decoded."""
    with urllib.request.urlopen(url, timeout=10) as r:
        buf = b""
        while True:
            # read1: whatever has arrived. read(n) waits for all n bytes,
            # which holds back a frame that is already here.
            chunk = r.read1(65536)
            if not chunk:
                return
            buf += chunk
            while True:
                a = buf.find(b"\xff\xd8")
                b = buf.find(b"\xff\xd9", a + 2) if a >= 0 else -1
                if a < 0 or b < 0:
                    break
                img = cv2.imdecode(np.frombuffer(buf[a:b + 2], np.uint8), cv2.IMREAD_COLOR)
                buf = buf[b + 2:]
                if img is not None:
                    yield img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--fullscreen", action="store_true")
    args = ap.parse_args()
    # GUI_NORMAL: no Qt toolbar or status bar, just the picture.
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    full = args.fullscreen
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
    sized = False
    while True:
        try:
            for img in frames(f"{args.base}/api/stream/video"):
                if not sized and not full:
                    cv2.resizeWindow(WINDOW, img.shape[1], img.shape[0])
                    sized = True
                cv2.imshow(WINDOW, img)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return
                if key == ord("f"):
                    full = not full
                    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    return                      # window closed with its X button
        except (OSError, http.client.HTTPException) as e:   # server down or restarted
            print(f"stream unavailable ({e}); retrying in 2 s - is the server running?")
            time.sleep(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
