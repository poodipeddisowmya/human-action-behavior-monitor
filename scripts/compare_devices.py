"""Laptop CPU vs Tesla T4 vs this device, on the same clips and software.

    python scripts/compare_devices.py dist/jetson-showcase

Left two columns come from the recording shipped in data/showcase/showcase.json
("before" = laptop CPU, "clips" = Kaggle Tesla T4); the third from a
recording made with scripts/record_showcase.py. Every figure uses the
scoring rules of scripts/build_showcase.py (what the screen showed during
12 s of each clip). Writes <folder>/comparison.md and comparison.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_showcase import matches  # noqa: E402

CROWD = ("office-people", "store-aisle", "street-surveillance")


def share(r: dict) -> float:
    return (r.get("labels_shown") or {}).get(r["expected"], 0.0)


def first(r: dict) -> float | None:
    return (r.get("first_seen_s") or {}).get(r["expected"])


def median(xs) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def by_stem(rows: list[dict]) -> dict[str, dict]:
    return {Path(r["video"]).stem: r for r in rows}


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "clips": n,
        "clear": sum(share(r) >= 0.5 for r in rows),
        "correct_top": sum(matches(r["expected"], r.get("labels_shown") or {}) for r in rows),
        "share_mean": sum(map(share, rows)) / n if n else 0.0,
        "appeared": sum(first(r) is not None for r in rows),
        "first_median_s": median(map(first, rows)),
        "fps_median": median(r.get("processing_fps") for r in rows),
    }


def fmt(v, kind: str) -> str:
    if v is None:
        return "—"
    return {"pct": f"{v:.0%}", "s": f"{v:.1f} s", "fps": f"{v:.1f}"}.get(kind, str(v))


def main() -> None:
    # Each argument: <folder> or <folder>=<column name>. The report is
    # written into the first folder.
    args = sys.argv[1:] or [str(ROOT / "dist" / "jetson-showcase")]
    ref = json.loads((ROOT / "data" / "showcase" / "showcase.json").read_text())
    sides = [("Laptop CPU", by_stem(ref.get("before", []))),
             (ref.get("gpu") or "GPU", by_stem(ref.get("clips", [])))]
    devs = []
    for arg in args:
        path, _, label = arg.partition("=")
        dev = json.loads((Path(path) / "showcase.json").read_text())
        name = label or (dev.get("gpu") or "This device").replace(
            "Engineering Reference Developer Kit ", "")
        sides.append((name, by_stem(dev.get("clips", []))))
        devs.append(dev)
    folder, dev = Path(args[0].partition("=")[0]), devs[0]
    names = [s[0] for s in sides]

    # Labelled clips recorded on ALL three sides: the only fair basis.
    labelled = sorted(set.intersection(*[
        {k for k, r in rows.items() if r.get("expected")} for _, rows in sides]))
    summ = [summarise([rows[c] for c in labelled]) for _, rows in sides]

    out = [f"# {' vs '.join(names)}", "",
           f"Same clips, same software, {ref.get('seconds_per_clip', 12)} s per clip; "
           f"every figure is what the screen showed. {len(labelled)} labelled clips "
           f"recorded on every device. Jetson recorded {dev.get('recorded')}"
           + (f", power mode {dev['power_mode']}" if dev.get("power_mode") else "") + ".", "",
           "| | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    rows = [("Clearly recognised (correct label on screen ≥ half the time)",
             [f"{s['clear']}/{s['clips']}" for s in summ]),
            ("Correct activity was the main label shown",
             [f"{s['correct_top']}/{s['clips']}" for s in summ]),
            ("Correct label on screen, average", [fmt(s["share_mean"], "pct") for s in summ]),
            ("Clips where the correct label appeared at all",
             [f"{s['appeared']}/{s['clips']}" for s in summ]),
            ("…how soon it appeared (median)", [fmt(s["first_median_s"], "s") for s in summ]),
            ("Playback speed, frames per second (median)",
             [fmt(s["fps_median"], "fps") for s in summ])]
    out += [f"| {k} | " + " | ".join(v) + " |" for k, v in rows]

    out += ["", "## Per clip", "",
            "Correct label's share of screen time · seconds until it first appeared · FPS.", "",
            "| Clip | Expected | " + " | ".join(names) + " |",
            "|---|---|" + "---|" * len(names)]
    per_clip = {}
    for c in labelled:
        cells = []
        for _, rows_ in sides:
            r = rows_[c]
            ok = "✓" if share(r) >= 0.5 else "✗"
            cells.append(f"{ok} {share(r):.0%} · {fmt(first(r), 's')} · {r.get('processing_fps', 0):.0f}")
        per_clip[c] = {n: {"share": share(rows_[c]), "first_s": first(rows_[c]),
                           "fps": rows_[c].get("processing_fps")}
                       for n, (_, rows_) in zip(names, sides)}
        out.append(f"| {c} | {sides[1][1][c]['expected']} | " + " | ".join(cells) + " |")

    crowd = [c for c in CROWD if all(c in rows for _, rows in sides)]
    if crowd:
        out += ["", "## Crowd scenes", "",
                "People tracked per frame (mean / max) · FPS.", "",
                "| Clip | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for c in crowd:
            out.append(f"| {c} | " + " | ".join(
                f"{rows[c].get('tracked_mean', 0):.1f} / {rows[c].get('tracked_max', 0)} · "
                f"{rows[c].get('processing_fps', 0):.0f}" for _, rows in sides) + " |")

    (folder / "comparison.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    (folder / "comparison.json").write_text(json.dumps({
        "devices": names, "labelled_clips": labelled,
        "summary": dict(zip(names, summ)), "per_clip": per_clip,
        "device_settings": [d.get("settings") for d in devs]}, indent=2))
    print("\n".join(out[:4 + len(rows) + 4]))
    print(f"\n-> {folder / 'comparison.md'}")


if __name__ == "__main__":
    main()
