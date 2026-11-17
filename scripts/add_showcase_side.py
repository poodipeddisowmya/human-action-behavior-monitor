"""Add another device's recording to the dashboard's comparison tab.

    python scripts/add_showcase_side.py dist/jetson-optimised --key jetson \\
        --tag "EDGE GPU" --sub "Jetson Orin Nano Super (8 GB)" --short Jetson

The tab compares the laptop CPU and the Tesla T4 recording installed by
scripts/install_showcase.py. This adds a third (or fourth...) side from a
folder written by scripts/record_showcase.py: the same clips through the same
software on another device. Its videos go to data/showcase/<key>/ and its
results into data/showcase/showcase.json under "extra_sides"; running it again
with the same key replaces that side. Run install_showcase.py first: it starts
data/showcase/ afresh.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "data" / "showcase"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--key", required=True, help="short id, also its folder: a-z, 0-9, -")
    ap.add_argument("--tag", required=True, help='badge on the video, e.g. "EDGE GPU"')
    ap.add_argument("--sub", required=True, help="device name shown next to the badge")
    ap.add_argument("--short", help="short name for the totals, e.g. Jetson (default: the key)")
    args = ap.parse_args()
    if not re.fullmatch(r"[a-z0-9-]+", args.key) or args.key in ("before", "left", "right"):
        sys.exit(f"bad key {args.key!r}: use a-z, 0-9 and -, not before/left/right")

    target = DST / "showcase.json"
    if not target.exists():
        sys.exit("no comparison installed: run scripts/install_showcase.py first")
    data = json.loads(target.read_text(encoding="utf-8"))
    dev = json.loads((args.folder / "showcase.json").read_text(encoding="utf-8"))

    # Only clips the installed CPU/T4 comparison also has.
    have = {Path(r["video"]).stem for r in data.get("clips", [])}
    rows = [r for r in dev.get("clips", []) if Path(r["video"]).stem in have
            and r.get("annotated_video")]
    out = DST / args.key
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    kept = []
    for r in rows:
        src = args.folder / Path(r["annotated_video"]).name
        if not src.is_file():
            print(f"  skip {Path(r['video']).stem}: no video")
            continue
        shutil.copy(src, out / src.name)
        kept.append({k: v for k, v in r.items() if k != "events"}
                    | {"annotated_video": src.name})

    side = {"key": args.key, "tag": args.tag, "sub": args.sub,
            "short": args.short or args.key.capitalize(),
            "gpu": dev.get("gpu"), "power_mode": dev.get("power_mode"),
            "recorded": dev.get("recorded"), "settings": dev.get("settings"),
            "clips": kept}
    data["extra_sides"] = [x for x in data.get("extra_sides", [])
                           if x.get("key") != args.key] + [side]
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    size = sum(f.stat().st_size for f in out.glob("*.mp4")) / 2**20
    print(f"  added {args.key}: {len(kept)} of {len(have)} clips ({size:.0f} MB) into {out}")


if __name__ == "__main__":
    main()
