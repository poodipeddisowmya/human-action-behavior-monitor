"""Teach the monitor new activities from your own video clips.

Put example clips in one folder per activity, named as the activity should
appear on screen, and run this. Full guide: docs/TRAINING.md.

    data/training/
        sleeping at desk/        one folder per new activity
            clip-001.mp4         2-30 s each, the activity visible throughout
            ...                  at least 8 clips; 20-50 from your own cameras is better
            phrase.txt           optional: subtitle wording, e.g. "asleep at their desk"
        other/                   optional, recommended: ordinary activity on the
                                 same cameras (desk work, walking, talking...)

    python scripts/train_activities.py --check      # check the clips only
    python scripts/train_activities.py              # train, measure, install
    python scripts/train_activities.py --backend both
    python scripts/train_activities.py --remove     # back to no trained activities

What it does:
  1. checks every clip (readable, long enough, a person visible);
  2. cuts each into 16-frame looks exactly as the live monitor does (whole
     frame for one person, a padded crop around the main person otherwise),
     plus mirrored copies, and runs them through the activity model
     (features cached in data/training/.cache/, so re-runs are quick);
  3. learns a small layer on those features, with an "other" class from
     the other/ folder and the bundled demo clips (data/samples/), so that
     ordinary activity is not mistaken for a new one;
  4. measures it on clips it was not trained on (cross-validation), picks
     the threshold that raises no false alarm on the "other" clips, and
     prints the report;
  5. installs it in models/custom_activities/ (the previous model is kept
     in models/custom_activities/backup/). Restart the monitor to use it.

Stop the monitor first on a Jetson: the activity model needs the GPU memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from config.settings import is_jetson  # noqa: E402
from src.action_recognition.clip_encoder import BACKENDS, CLIP_LEN, prepare_frame  # noqa: E402
from src.action_recognition.custom_activities import (  # noqa: E402
    OTHER, ActivityHead, confirm_rule, model_dir, model_paths, now_iso, strong_for, train_head)

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MIN_CLIPS = 8            # per activity, to train at all
GOOD_CLIPS = 20          # below this, a warning
MAX_FPS = 30             # faster video is sampled down to about this
WINDOW_STRIDE = 8        # frames between looks, as live (_STRIDE_FRAMES)
MAX_WINDOWS = 12         # looks per clip, evenly spread: long clips must not dominate
MIN_FRAMES = CLIP_LEN + WINDOW_STRIDE
CACHE_VERSION = 1
THRESHOLDS = [round(t, 2) for t in np.arange(0.5, 0.96, 0.05)] + [0.97, 0.98, 0.99]
# Regularisation strength, chosen on held-out clips by what matters: the
# most activity clips recognised with no false alarm on "other" (measured on
# the example: 0.1 for VideoMAE, 0.3 for S3D; stronger or weaker cost recall).
WEIGHT_DECAYS = [0.01, 0.03, 0.1, 0.3, 1.0]
_NAME = re.compile(r"^[a-z0-9][a-z0-9 '\-]{0,38}[a-z0-9]$")


def say(msg: str = "") -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. The clips
# ---------------------------------------------------------------------------
def activity_name(folder: Path) -> str:
    return re.sub(r"\s+", " ", folder.name.replace("_", " ")).strip().lower()


def videos_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT
                  and not any(part.startswith(".") for part in p.relative_to(folder).parts))


def probe(path: Path) -> str | None:
    """Why this clip cannot be used, or None if it can."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return "cannot be opened (damaged, or an unsupported codec)"
        ok, _ = cap.read()
        if not ok:
            return "no frames could be read"
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, round(fps / MAX_FPS))
        if n and n // step < MIN_FRAMES:
            return f"too short ({n / fps:.1f} s; at least {MIN_FRAMES * step / fps:.1f} s)"
    finally:
        cap.release()
    return None


def discover(data_dir: Path, negative_dirs: list[Path]):
    """{activity: [clips]}, [other clips], {activity: phrase}, [problems]."""
    problems: list[str] = []
    classes: dict[str, list[Path]] = {}
    phrases: dict[str, str] = {}
    other: list[Path] = []
    if not data_dir.is_dir():
        raise SystemExit(f"No training folder at {data_dir}. Create it with one folder per "
                         f"activity inside (docs/TRAINING.md).")
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        name = activity_name(folder)
        clips = videos_in(folder)
        if name == OTHER:
            other += clips
            continue
        if not _NAME.match(name):
            problems.append(f"folder '{folder.name}': use 2-40 letters, digits, spaces, ' or -")
            continue
        classes[name] = clips
        phrase_file = folder / "phrase.txt"
        if phrase_file.is_file():
            phrases[name] = phrase_file.read_text(encoding="utf-8").strip()[:60] or name
    if not classes:
        raise SystemExit(f"No activity folders in {data_dir} (docs/TRAINING.md).")
    training = {p.resolve() for clips in classes.values() for p in clips}
    for d in negative_dirs:
        if d.is_dir():
            other += [p for p in videos_in(d) if p.resolve() not in training]
    other = sorted(set(other))
    return classes, other, phrases, problems


# ---------------------------------------------------------------------------
# 2. Looks -> features, as live
# ---------------------------------------------------------------------------
def _cache_file(path: Path, backend: str, detector: str) -> Path:
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{backend}|{detector}|{CACHE_VERSION}|{MAX_WINDOWS}"
    return ROOT / "data" / "training" / ".cache" / backend / f"{hashlib.sha1(key.encode()).hexdigest()}.npz"


def clip_features(path: Path, encoder, detector) -> tuple[np.ndarray, int]:
    """Features of up to MAX_WINDOWS looks at this clip, each also mirrored:
    (n, dim) in time order (each look followed by its mirror), and the
    number of looks skipped because nobody was visible."""
    cache = _cache_file(path, encoder.backend, detector.weights_name)
    if cache.is_file():
        with np.load(cache) as z:
            return z["feats"], int(z["skipped"])
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps / MAX_FPS))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) // step
    starts = list(range(0, max(0, n - CLIP_LEN) + 1, WINDOW_STRIDE)) or [0]
    if len(starts) > MAX_WINDOWS:
        starts = [starts[i] for i in np.linspace(0, len(starts) - 1, MAX_WINDOWS).astype(int)]
    wanted = {s + CLIP_LEN - 1: s for s in starts}          # window's last frame -> start
    frames: deque = deque(maxlen=CLIP_LEN)
    feats, skipped, i, raw = [], 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw += 1
        if (raw - 1) % step:
            continue
        frames.append(frame)
        if i in wanted and len(frames) == CLIP_LEN:
            boxes = detector.detect(frames[CLIP_LEN // 2])
            if not boxes:
                skipped += 1                    # live, nobody in view = no look at all
            else:
                main = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                single = len(boxes) == 1
                clip = [prepare_frame(f, main, single) for f in frames]
                if all(c is not None for c in clip):
                    feats.append(encoder.forward(clip)[1])
                    feats.append(encoder.forward([cv2.flip(c, 1) for c in clip])[1])
        i += 1
    cap.release()
    arr = np.asarray(feats, np.float32).reshape(-1, encoder.dim)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, feats=arr, skipped=skipped)
    return arr, skipped


# ---------------------------------------------------------------------------
# 3-4. Train and measure on held-out clips
# ---------------------------------------------------------------------------
def live_rule_fires(p_seq: np.ndarray, cls: int, thr: float, rule: tuple | None = None) -> bool:
    """The live confirmation rule (custom_activities.confirm_rule) on a
    clip's looks in time order: `min_hits` of the last `looks` looks at or
    above the threshold, or one look beyond strong_for(threshold)."""
    min_hits, window, strong = rule or confirm_rule()
    # A look counts for an activity only when it is that look's top class,
    # as live.
    top = p_seq.argmax(axis=1)
    p = np.where(top == cls, p_seq[:, cls], 0.0)
    if (p >= strong_for(thr, strong)).any():
        return True
    hits = p >= thr
    return any(hits[max(0, k - window + 1):k + 1].sum() >= min_hits for k in range(len(hits)))


def cross_validate(clips: list[tuple[np.ndarray, int]], classes: list[str], backend: str,
                   folds: int, seed: int, weight_decay: float = 1e-3) -> dict:
    """Held-out probabilities for every clip: each clip is scored by a layer
    trained without it (stratified k-fold over clips, not looks)."""
    rng = np.random.default_rng(seed)
    fold_of = np.zeros(len(clips), int)
    for c in range(len(classes)):
        idx = [i for i, (_, y) in enumerate(clips) if y == c]
        rng.shuffle(idx)
        for k, i in enumerate(idx):
            fold_of[i] = k % folds
    held: dict[int, np.ndarray] = {}
    for f in range(folds):
        train = [i for i in range(len(clips)) if fold_of[i] != f]
        test = [i for i in range(len(clips)) if fold_of[i] == f]
        if not test:
            continue
        x = np.concatenate([clips[i][0] for i in train])
        y = np.concatenate([np.full(len(clips[i][0]), clips[i][1]) for i in train])
        head = train_head(x, y, classes, backend, seed=seed, weight_decay=weight_decay)
        for i in test:
            held[i] = head.probs(clips[i][0][0::2])             # originals only, in time order
    return held


def evaluate(clips, held, classes: list[str]) -> dict:
    """Per threshold: activity clips recognised, and false alarms on "other"."""
    other = classes.index(OTHER)
    acts = [c for c in range(len(classes)) if c != other]
    out = {}
    for thr in THRESHOLDS:
        per = {}
        for c in acts:
            idx = [i for i, (_, y) in enumerate(clips) if y == c and i in held]
            got = sum(live_rule_fires(held[i], c, thr) for i in idx)
            wrong = sum(any(live_rule_fires(held[i], d, thr) for d in acts if d != c) for i in idx)
            per[classes[c]] = {"clips": len(idx), "recognised": got, "confused": wrong}
        oidx = [i for i, (_, y) in enumerate(clips) if y == other and i in held]
        alarms = sum(any(live_rule_fires(held[i], c, thr) for c in acts) for i in oidx)
        out[thr] = {"activities": per, "other_clips": len(oidx), "false_alarm_clips": alarms}
    return out


def choose_threshold(table: dict) -> float:
    """The lowest threshold with no false alarm on the "other" clips (best
    recall that keeps ordinary activity quiet); else the one with fewest."""
    clean = [t for t in THRESHOLDS if table[t]["false_alarm_clips"] == 0]
    if clean:
        return clean[0]
    return min(THRESHOLDS, key=lambda t: (table[t]["false_alarm_clips"], t))


# ---------------------------------------------------------------------------
# 5. Install
# ---------------------------------------------------------------------------
def install(head: ActivityHead, out_dir: Path | None) -> tuple[Path, Path]:
    npz, meta = model_paths(head.backend)
    if out_dir is not None:
        npz, meta = out_dir / npz.name, out_dir / meta.name
    elif npz.exists():
        backup = model_dir() / "backup"
        backup.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for f in (npz, meta):
            if f.exists():
                shutil.copy2(f, backup / f"{f.stem}-{stamp}{f.suffix}")
    head.save(npz, meta)
    return npz, meta


def monitor_running() -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{settings.api_port}/api/health", timeout=2)
        return True
    except Exception:
        return False


def remove() -> int:
    removed = False
    for b in BACKENDS:
        for f in model_paths(b):
            if f.exists():
                backup = model_dir() / "backup"
                backup.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), backup / f"{f.stem}-removed-{datetime.now():%Y%m%d-%H%M%S}{f.suffix}")
                removed = True
    say("Trained activities removed (copies kept in models/custom_activities/backup/). "
        "Restart the monitor." if removed else "No trained activities installed.")
    return 0


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "training")
    ap.add_argument("--negatives", type=Path, nargs="*", default=[ROOT / "data" / "samples"],
                    help="more folders of ordinary activity for the 'other' class "
                         "(default: the bundled demo clips)")
    ap.add_argument("--backend", default=settings.model.deep_backend,
                    choices=[*BACKENDS, "both"],
                    help=f"activity model to train for (default: this machine's, "
                         f"{settings.model.deep_backend})")
    ap.add_argument("--folds", type=int, default=4, help="cross-validation folds")
    ap.add_argument("--threshold", type=float, default=None, help="use this instead of choosing one")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write the model here instead of installing it")
    ap.add_argument("--check", action="store_true", help="check the clips, train nothing")
    ap.add_argument("--remove", action="store_true", help="uninstall the trained activities")
    ap.add_argument("--force", action="store_true", help="train even if the monitor is running")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.remove:
        return remove()

    say(f"Training folder: {args.data}")
    classes, other, phrases, problems = discover(args.data, args.negatives)
    say("Checking clips...")
    usable: dict[str, list[Path]] = {}
    for name, clips in list(classes.items()) + [(OTHER, other)]:
        ok = []
        for p in clips:
            why = probe(p)
            if why:
                problems.append(f"{p.relative_to(ROOT) if ROOT in p.parents else p}: {why}")
            else:
                ok.append(p)
        usable[name] = ok
    errors = []
    say()
    for name in classes:
        n = len(usable[name])
        note = ("not enough: at least %d" % MIN_CLIPS) if n < MIN_CLIPS else (
            "ok; %d+ from your own cameras is better" % GOOD_CLIPS if n < GOOD_CLIPS else "good")
        say(f"  {name:<28} {n:>3} clips   {note}")
        if n < MIN_CLIPS:
            errors.append(f"'{name}' has {n} usable clips; at least {MIN_CLIPS} are needed")
    say(f"  {'(other: ordinary activity)':<28} {len(usable[OTHER]):>3} clips   "
        + ("ok" if len(usable[OTHER]) >= MIN_CLIPS else "too few: add clips to other/"))
    if len(usable[OTHER]) < MIN_CLIPS:
        errors.append(f"only {len(usable[OTHER])} 'other' clips; add ordinary activity to "
                      f"{args.data / OTHER}")
    for pr in problems:
        say(f"  ! skipped {pr}")
    if errors:
        say("\nCannot train yet:\n  - " + "\n  - ".join(errors))
        return 1
    if args.check:
        say("\nAll set: run without --check to train.")
        return 0
    if monitor_running() and is_jetson() and not args.force:
        say(f"\nThe monitor is running (port {settings.api_port}). On a Jetson it holds the GPU "
            f"memory the activity model needs: stop it, then run this again (or --force).")
        return 1

    from src.action_recognition.clip_encoder import ClipEncoder
    from src.tracking import PersonTracker
    names = sorted(classes) + [OTHER]
    detector = PersonTracker()
    backends = list(BACKENDS) if args.backend == "both" else [args.backend]
    for backend in backends:
        say(f"\n=== {backend} ===")
        t0 = time.time()
        encoder = ClipEncoder(backend)
        data: list[tuple[np.ndarray, int]] = []
        no_person: list[str] = []
        total = sum(len(v) for v in usable.values())
        done = 0
        for name in names:
            for p in usable[name]:
                feats, skipped = clip_features(p, encoder, detector)
                done += 1
                if len(feats) == 0:
                    no_person.append(p.name)
                else:
                    data.append((feats, names.index(name)))
                if done % 10 == 0 or done == total:
                    say(f"  looked at {done}/{total} clips ({time.time() - t0:.0f} s)")
        if no_person:
            say(f"  ! no person found in {len(no_person)} clip(s), not used: "
                + ", ".join(no_person[:5]) + (" ..." if len(no_person) > 5 else ""))
        counts = {n: sum(1 for _, y in data if y == names.index(n)) for n in names}
        short = [n for n in names if counts[n] < 2]
        if short:
            say(f"Cannot train: no usable clips left for {', '.join(short)}.")
            return 1
        folds = max(2, min(args.folds, min(counts.values())))
        say(f"  measuring on held-out clips ({folds}-fold cross-validation)...")
        best = None
        for wd in WEIGHT_DECAYS:
            h = cross_validate(data, names, backend, folds, args.seed, wd)
            t = evaluate(data, h, names)
            th = args.threshold if args.threshold is not None else choose_threshold(t)
            r = t[min(THRESHOLDS, key=lambda x: abs(x - th))]
            score = (r["false_alarm_clips"], -sum(v["recognised"] for v in r["activities"].values()))
            if best is None or score < best[0]:
                best = (score, wd, t, th)
        _, weight_decay, table, thr = best
        res = table[min(THRESHOLDS, key=lambda t: abs(t - thr))]

        mh, lk, _ = confirm_rule()
        say(f"\n  Held-out clips at threshold {thr:.2f} (live rule: {mh} of the last {lk} looks):")
        for n, r in res["activities"].items():
            say(f"    {n:<28} recognised in {r['recognised']:>3}/{r['clips']:<3} clips"
                + (f"   (mistaken for another activity in {r['confused']})" if r["confused"] else ""))
        say(f"    {'other (ordinary activity)':<28} false alarm in {res['false_alarm_clips']:>3}/"
            f"{res['other_clips']} clips")
        n_act = sum(r["clips"] for r in res["activities"].values())
        say("\n  Other thresholds (--threshold T to choose one; higher = fewer false alarms):")
        say("    threshold  recognised  false alarms")
        for t in THRESHOLDS:
            v = table[t]
            mark = "  <- chosen" if abs(t - thr) < 1e-9 else ""
            say(f"      {t:.2f}     {sum(x['recognised'] for x in v['activities'].values()):>3}/{n_act:<3}"
                f"      {v['false_alarm_clips']:>3}/{v['other_clips']}{mark}")
        if res["false_alarm_clips"]:
            say("  ! Some ordinary clips still trigger it: add clips like them to other/ and retrain.")
        recall = sum(r["recognised"] for r in res["activities"].values()) / max(
            1, sum(r["clips"] for r in res["activities"].values()))
        if recall < 0.7:
            say("  ! Recognised in under 70% of held-out clips: add more, and more varied, clips.")

        x = np.concatenate([f for f, _ in data])
        y = np.concatenate([np.full(len(f), c) for f, c in data])
        head = train_head(x, y, names, backend, seed=args.seed, weight_decay=weight_decay)
        head.threshold = float(thr)
        head.meta = {
            "trained_at": now_iso(),
            "phrases": {n: phrases.get(n, n) for n in names if n != OTHER},
            "clips": counts,
            "looks": int(len(x)),
            "data": str(args.data.resolve().relative_to(ROOT)) if ROOT in args.data.resolve().parents
                    else args.data.name,
            "validation": {
                "folds": folds, "threshold": thr, "weight_decay": weight_decay,
                "rule": dict(zip(("min_hits", "looks", "strong"), confirm_rule())),
                "clip_accuracy": round(recall, 3),
                "activities": res["activities"],
                "other_clips": res["other_clips"],
                "false_alarm_clips": res["false_alarm_clips"],
                "by_threshold": {str(t): {"recognised": sum(r["recognised"] for r in v["activities"].values()),
                                          "false_alarms": v["false_alarm_clips"]}
                                 for t, v in table.items()},
            },
        }
        npz, meta = install(head, args.out_dir)
        say(f"\n  Saved {npz.relative_to(ROOT) if ROOT in npz.parents else npz} "
            f"({time.time() - t0:.0f} s).")
        del encoder
    if args.out_dir is None:
        say("\nRestart the monitor to use the trained activities.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
