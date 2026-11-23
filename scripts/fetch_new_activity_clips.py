"""Download the demo clips for the new activities (making tea, taking tablets,
falling down) from their original sources into data/samples/.

They are not in the repository: the stock footage's licences let anyone
download and use it, but not re-share the raw files. This fetches each one
from its source, converts it to .mp4 where needed, and names it so the
dashboard's Overview and Source menu pick it up (new-*.mp4, git-ignored).
Attribution: data/showcase/CREDITS.md.

    python scripts/fetch_new_activity_clips.py            # missing clips only
    python scripts/fetch_new_activity_clips.py --force    # download again
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]

# (file in data/samples, source file URL, page to download it by hand,
#  what it shows, licence)
_COMMONS = "https://commons.wikimedia.org/wiki/File:"
CLIPS = [
    ("new-taking-tablets-1.mp4", "https://assets.mixkit.co/videos/46792/46792-720.mp4",
     "https://mixkit.co/free-stock-video/elderly-man-takes-pills-to-control-his-illnesses-46792/",
     "elderly man takes pills with water", "Mixkit Free License"),
    ("new-taking-tablets-2.mp4", "https://assets.mixkit.co/videos/48203/48203-720.mp4",
     "https://mixkit.co/free-stock-video/young-woman-taking-pills-48203/",
     "young woman takes pills with water", "Mixkit Free License"),
    ("new-falling-down.mp4", "https://assets.mixkit.co/videos/23814/23814-720.mp4",
     "https://mixkit.co/free-stock-video/tired-man-falls-in-the-snow-23814/",
     "man falls in the snow", "Mixkit Free License"),
    ("new-making-tea-1.mp4",
     "https://upload.wikimedia.org/wikipedia/commons/f/fe/Making_chai_in_Kumbakonam.ogv",
     _COMMONS + "Making_chai_in_Kumbakonam.ogv", "man making chai at a stall", "CC BY 2.0"),
    ("new-making-tea-2.mp4",
     "https://upload.wikimedia.org/wikipedia/commons/7/7f/Traditional_Tea_Making_01.webm",
     _COMMONS + "Traditional_Tea_Making_01.webm", "tea brewed in a large urn", "CC BY-SA 4.0"),
    ("new-making-tea-3.mp4",
     "https://upload.wikimedia.org/wikipedia/commons/5/59/How_Nomads_prepare_Moroccan_tea.webm",
     _COMMONS + "How_Nomads_prepare_Moroccan_tea.webm", "Moroccan tea prepared outdoors",
     "CC BY-SA 4.0"),
]

# A file downloaded by hand goes here, under its original name; the script
# converts it instead of downloading (git-ignored).
MANUAL_DIR = ROOT / "data" / "new-activities"

# Wikimedia asks automated clients for a descriptive User-Agent. It can still
# rate-limit a busy network (HTTP 429): the script then waits as told, once,
# and otherwise says where to download the file by hand.
_UA = "human-action-behavior-monitor/1.0 (demo clip fetcher)"
_MAX_WAIT_S = 130


class Busy(Exception):
    """The server keeps refusing for now (rate limit)."""


def _download(url: str, dest: Path) -> None:
    """Download; on a rate limit (429/503) wait as the server asks, once."""
    import time
    import urllib.error
    for attempt in range(2):
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            return
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503):
                raise
            wait = int(e.headers.get("Retry-After", "0") or 0) or 30
            if attempt == 1 or wait > _MAX_WAIT_S:
                raise Busy(f"the server is rate-limiting this network (HTTP {e.code})") from e
            print(f"         server busy: waiting {wait} s as it asks", flush=True)
            time.sleep(wait)


def _to_mp4(src: Path, dest: Path) -> int:
    """Re-encode any video OpenCV can read to .mp4; returns frames written."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError("cannot read the downloaded video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = fps if 1 <= fps <= 60 else 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = dest.with_suffix(".part.mp4")
    out = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
        n += 1
    cap.release()
    out.release()
    if n == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("no frames decoded")
    tmp.replace(dest)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true", help="download clips already present")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "samples")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    failed = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, url, page, what, licence in CLIPS:
            dest = args.out / name
            if dest.exists() and not args.force:
                print(f"  have   {name}")
                continue
            print(f"  fetch  {name}  ({what}, {licence})", flush=True)
            manual = MANUAL_DIR / Path(url).name
            raw = manual if manual.is_file() else Path(tmp) / ("raw" + Path(url).suffix)
            try:
                if raw is not manual:
                    _download(url, raw)
                n = _to_mp4(raw, dest)
                print(f"         {n} frames, {dest.stat().st_size // 1024} KB")
            except Exception as e:
                failed.append((name, page, Path(url).name, e))
                print(f"  FAILED {name}: {e}")
    print(f"\n{len(CLIPS) - len(failed)} of {len(CLIPS)} clips ready in {args.out}.")
    if failed:
        print("\nFor the others, run this again later, or download each by hand and run it again:")
        for name, page, original, _ in failed:
            print(f"  {page}\n      save as  {(MANUAL_DIR / original).relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
