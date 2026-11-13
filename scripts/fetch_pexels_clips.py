"""Fetch realistic, production-quality demo clips from Pexels.

Why Pexels and not YouTube
--------------------------
YouTube's Terms of Service prohibit downloading, and the videos are
copyrighted regardless of who does the downloading — building a client demo
on scraped YouTube footage creates real legal exposure for whoever ships it.

Pexels video is licensed for free use including commercial use, with no
attribution required. The footage is professionally shot, HD, and realistic —
which is exactly what makes it better demo material than low-resolution
research datasets like UCF101/HMDB51.

Setup (about 30 seconds, free)
------------------------------
    1. https://www.pexels.com/api/  ->  sign up  ->  copy your API key
    2. set it:      setx PEXELS_API_KEY "your_key_here"      (new shell after)
       or pass it:  python scripts/fetch_pexels_clips.py --key your_key_here

Usage
-----
    python scripts/fetch_pexels_clips.py                 # all categories
    python scripts/fetch_pexels_clips.py --only cooking eating
    python scripts/fetch_pexels_clips.py --per-category 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

API = "https://api.pexels.com/videos/search"

# Search terms tuned per target category. Multiple phrasings per category
# because Pexels' relevance varies a lot with wording — the first query that
# returns a usable clip wins.
QUERIES: dict[str, list[str]] = {
    "cooking":        ["person cooking kitchen", "chef chopping vegetables", "woman cooking food"],
    "eating":         ["person eating meal", "man eating food restaurant", "woman eating"],
    "drinking":       ["person drinking water glass", "man drinking coffee", "woman drinking"],
    "smoking":        ["person smoking cigarette", "man smoking outdoors", "woman smoking"],
    "reading":        ["person reading book", "woman reading indoors", "man reading newspaper"],
    "writing":        ["person writing notebook", "woman writing desk", "hand writing paper"],
    "using-computer": ["person typing laptop", "woman working computer", "man using laptop desk"],
    "using-phone":    ["person using smartphone", "woman texting phone", "man looking at phone"],
    "cleaning":       ["person mopping floor", "woman cleaning house", "person vacuuming"],
    "exercising":     ["man doing pushups", "woman exercising gym", "person doing squats"],
    "cycling":        ["person riding bicycle", "man cycling road", "woman riding bike"],
    "driving":        ["person driving car", "man driving vehicle", "driver steering wheel"],
}

# Prefer a file that is big enough to see detail but small enough to loop
# smoothly and download fast.
TARGET_W = 960
MAX_BYTES = 40 * 1024 * 1024


def _get(url: str, key: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": key})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def _pick_file(video: dict) -> dict | None:
    """Choose the video file closest to TARGET_W without going huge."""
    files = [f for f in video.get("video_files", [])
             if f.get("file_type") == "video/mp4" and f.get("width")]
    if not files:
        return None
    files.sort(key=lambda f: abs(f["width"] - TARGET_W))
    return files[0]


def fetch_category(cat: str, key: str, out_dir: Path, count: int) -> int:
    saved = 0
    for query in QUERIES[cat]:
        if saved >= count:
            break
        url = (f"{API}?query={urllib.parse.quote(query)}"
               f"&per_page=8&orientation=landscape&size=medium")
        try:
            data = _get(url, key)
        except Exception as e:
            print(f"    query failed ({query!r}): {str(e)[:80]}")
            continue

        for vid in data.get("videos", []):
            if saved >= count:
                break
            # Very short clips don't give the classifier a 16-frame window
            # across a stable track; very long ones bloat the repo.
            dur = vid.get("duration", 0)
            if not (5 <= dur <= 40):
                continue
            f = _pick_file(vid)
            if not f:
                continue

            suffix = "" if saved == 0 else f"-{saved + 1}"
            dest = out_dir / f"demo-{cat}{suffix}.mp4"
            if dest.exists() and dest.stat().st_size > 100_000:
                print(f"    ~ {dest.name} already present")
                saved += 1
                continue
            try:
                req = urllib.request.Request(
                    f["link"], headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=90) as r, dest.open("wb") as o:
                    total = 0
                    while chunk := r.read(1 << 20):
                        total += len(chunk)
                        if total > MAX_BYTES:
                            raise RuntimeError("file too large")
                        o.write(chunk)
                print(f"    + {dest.name:<28} {dest.stat().st_size // 1024:>6} KB"
                      f"  {f['width']}x{f['height']}  {dur}s")
                saved += 1
            except Exception as e:
                dest.unlink(missing_ok=True)
                print(f"    ! download failed: {str(e)[:70]}")
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=os.environ.get("PEXELS_API_KEY"))
    ap.add_argument("--only", nargs="*", help="subset of categories")
    ap.add_argument("--per-category", type=int, default=1)
    args = ap.parse_args()

    if not args.key:
        print(__doc__)
        sys.exit("ERROR: no API key. Set PEXELS_API_KEY or pass --key.")

    out = settings.data_dir / "samples"
    out.mkdir(parents=True, exist_ok=True)

    cats = args.only or list(QUERIES)
    unknown = [c for c in cats if c not in QUERIES]
    if unknown:
        sys.exit(f"unknown categories: {unknown}\navailable: {list(QUERIES)}")

    print(f"Fetching Pexels clips -> {out}\n")
    total = 0
    for cat in cats:
        print(f"  {cat}")
        total += fetch_category(cat, args.key, out, args.per_category)
    print(f"\n{total} clip(s) saved. They appear in the UI source dropdown "
          f"immediately — no restart needed.")


if __name__ == "__main__":
    main()
