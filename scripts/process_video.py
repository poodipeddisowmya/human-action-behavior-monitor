"""Process a video file end-to-end into an annotated video + an event report.

    python scripts/process_video.py data/samples/demo-cooking.mp4
    python scripts/process_video.py in.mp4 --out runs/report

Writes, in the output folder:
    <name>_annotated.mp4   boxes, IDs, actions and the caption burned in
                           (H.264 when ffmpeg is available, so it plays in a
                           browser or a notebook)
    <name>_events.json     every completed action: person, label, start, end,
                           duration, confidence
    <name>_summary.json    per-label totals, frame rate achieved, device

It runs the very same Pipeline the live dashboard runs — same models, same
smoothing, same event rules — so what it reports is what a viewer would have
seen. Playback is real-time by design (the deep model's pacing is
time-based), so a 30 s clip takes about 30 s.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

# config / the pipeline are imported where a clip is processed, not here.
# Importing config initialises CUDA, and the batch parent below only spawns
# one child per clip: holding an idle CUDA context there cost ~300 MB that a
# Jetson (GPU memory = system RAM) needed for the models in the child.


def _to_h264(src: Path) -> Path:
    """Re-encode to H.264 (mp4v does not play in browsers). Keeps src on failure."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return src
    dst = src.with_name(src.stem.replace("_raw", "") + ".mp4")
    r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)])
    if r.returncode == 0 and dst.exists():
        src.unlink(missing_ok=True)
        return dst
    return src


def process(video: Path, out_dir: Path, min_seconds: float = 0.0,
            max_width: int = 0) -> dict:
    """min_seconds > 0 loops a shorter clip until that much video is written,
    as the live dashboard loops it — a 4 s clip otherwise ends about when
    the activity label has just settled. max_width > 0 downscales the output
    (a 2732 px source makes an unwieldy file to send)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    from config import settings
    from src.input import FileSource
    from src.pipeline import Pipeline

    source = FileSource(video, loop=min_seconds > 0)
    fps = source.fps or 25.0
    target = int(min_seconds * fps) if min_seconds > 0 else None
    raw = out_dir / f"{video.stem}_annotated_raw.mp4"
    writer: list[cv2.VideoWriter] = []
    events: list[dict] = []
    shown: Counter = Counter()
    first_seen: dict[str, float] = {}       # label -> video second it first appeared
    last_label: dict[int, str] = {}         # person -> label on the previous frame
    changes = 0                             # on-screen label changes (flicker)
    tracked: list[int] = []
    frames = 0
    done = threading.Event()

    def on_frame(img, annotations, _caption=None):
        nonlocal frames, changes
        if done.is_set():
            return
        tracked.append(len(annotations))
        for a in annotations:
            lbl = str(a.get("label", ""))
            first_seen.setdefault(lbl, round(frames / fps, 2))
            prev = last_label.get(a["person_id"])
            if prev is not None and prev != lbl:
                changes += 1
            last_label[a["person_id"]] = lbl
        h, w = img.shape[:2]
        if max_width and w > max_width:
            img = cv2.resize(img, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        if not writer:
            writer.append(cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)))
        writer[0].write(img)
        for a in annotations:
            if not str(a.get("label", "")).startswith("detecting"):
                shown[a["label"]] += 1
        frames += 1
        if target and frames >= target:
            done.set()

    def on_event(e):
        if e.label.startswith("detecting"):
            return          # the "model hasn't looked yet" placeholder, not an action
        d = e.to_dict()
        d["person_id"] = f"P{e.person_id:03d}"
        events.append(d)

    pipeline = Pipeline(source, on_frame=on_frame, on_event=on_event)
    t0 = time.time()
    pipeline.start()
    # Ends when a one-shot file has been played, or once min_seconds of
    # video has been written for a looping one.
    while pipeline._thread.is_alive() and not done.wait(0.2):
        pass
    pipeline.stop()
    secs = time.time() - t0
    if writer:
        writer[0].release()
    video_out = _to_h264(raw) if raw.exists() else None

    per_label: dict[str, dict] = defaultdict(lambda: {"events": 0, "seconds": 0.0})
    for e in events:
        per_label[e["label"]]["events"] += 1
        per_label[e["label"]]["seconds"] += float(e.get("duration_s") or 0)
    summary = {
        "video": video.name,
        "device": settings.device,
        "frames": frames,
        "processing_fps": round(frames / secs, 1) if secs else 0,
        "people": len({e["person_id"] for e in events}),
        "events": len(events),
        "by_label": {k: {"events": v["events"], "seconds": round(v["seconds"], 1)}
                     for k, v in sorted(per_label.items(), key=lambda kv: -kv[1]["seconds"])},
        # Share of on-screen person-labels, i.e. what a viewer actually saw.
        "labels_shown": {k: round(n / max(1, sum(shown.values())), 3)
                         for k, n in shown.most_common(5)},
        "first_seen_s": first_seen,
        "label_changes_per_10s": round(changes / max(1e-6, frames / fps) * 10, 2),
        "tracked_mean": round(sum(tracked) / max(1, len(tracked)), 2),
        "tracked_max": max(tracked, default=0),
        "annotated_video": str(video_out) if video_out else None,
    }
    (out_dir / f"{video.stem}_events.json").write_text(json.dumps(events, indent=2, default=str))
    (out_dir / f"{video.stem}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs/processed"))
    ap.add_argument("--min-seconds", type=float, default=0.0,
                    help="loop shorter clips until this much video is written")
    ap.add_argument("--max-width", type=int, default=0,
                    help="downscale the annotated video to at most this width")
    args = ap.parse_args()
    if len(args.videos) > 1:
        # One process per clip. Each Pipeline loads its own models and its
        # worker threads keep them alive after stop(), so a long batch in one
        # process ran a T4 out of memory around the 30th clip (12.5 GB used;
        # later clips silently lost detection). A fresh process per clip
        # returns all of it.
        for v in args.videos:
            subprocess.run([sys.executable, __file__, str(v), "--out", str(args.out),
                            "--min-seconds", str(args.min_seconds),
                            "--max-width", str(args.max_width)])
        return
    v = args.videos[0]
    if not v.exists():
        print(f"missing: {v}")
        return
    s = process(v, args.out, args.min_seconds, args.max_width)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
