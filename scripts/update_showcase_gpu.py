"""Replace the GPU side of the dashboard's comparison with a new recording.

    python scripts/kaggle_auto.py showcase           # T4 recording -> dist/client-demo
    python scripts/update_showcase_gpu.py dist/client-demo

The comparison tab (data/showcase/showcase.json) holds three recordings made
at different times: the laptop CPU ("before"), the GPU ("clips") and extra
devices ("extra_sides", e.g. the Jetson). install_showcase.py replaces all of
it; this replaces only the GPU side, keeping the others, so the T4 can be
re-recorded with new clips (the new activities) and the current code.
Clips the CPU never recorded show as "not recorded" on that side.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "data" / "showcase"


def main(src: Path) -> None:
    target = DST / "showcase.json"
    if not target.exists():
        sys.exit("no comparison installed: run scripts/install_showcase.py first")
    data = json.loads(target.read_text(encoding="utf-8"))
    new = json.loads((src / "showcase.json").read_text(encoding="utf-8"))

    rows = []
    for r in new.get("clips", []):
        name = Path(r.get("annotated_video") or "").name
        if not name or not (src / name).is_file():
            print(f"  skip {Path(r['video']).stem}: no video")
            continue
        shutil.copy(src / name, DST / name)
        rows.append({k: v for k, v in r.items() if k != "events"} | {"annotated_video": name})

    # GPU videos no longer referenced (a clip dropped from the set) go.
    keep = {r["annotated_video"] for r in rows}
    for old in data.get("clips", []):
        f = DST / Path(old.get("annotated_video") or "").name
        if f.is_file() and f.name not in keep:
            f.unlink()

    data.update({"clips": rows, "gpu": new.get("gpu") or data.get("gpu"),
                 "recorded": new.get("recorded"), "minutes": new.get("minutes"),
                 "seconds_per_clip": new.get("seconds_per_clip", data.get("seconds_per_clip"))})
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    cpu = {Path(r["video"]).stem for r in data.get("before", [])}
    added = [Path(r["video"]).stem for r in rows if Path(r["video"]).stem not in cpu]
    print(f"  GPU side: {len(rows)} clips from {src} ({data['gpu']}, {data['recorded']})"
          + (f"; GPU-only: {', '.join(added)}" if added else ""))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "dist/client-demo"))
