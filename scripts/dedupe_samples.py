"""Remove duplicate clips from data/samples/.

Matches on CONTENT (sha256), not filename — re-uploading the same video
produced copies named foo.mp4, foo_1.mp4, foo_2.mp4 which are byte-identical.
The shortest filename is kept as canonical.

    python scripts/dedupe_samples.py            # dry run, shows what it would do
    python scripts/dedupe_samples.py --apply    # actually delete
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete")
    args = ap.parse_args()

    samples = settings.data_dir / "samples"
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(samples.iterdir()):
        if f.suffix.lower() in VIDEO_EXT and f.is_file():
            by_hash[sha256(f)].append(f)

    dupes = {h: fs for h, fs in by_hash.items() if len(fs) > 1}
    if not dupes:
        print(f"No duplicates among {sum(len(v) for v in by_hash.values())} clips.")
        return

    freed = 0
    removed = 0
    for h, files in dupes.items():
        # Keep the shortest name — the original, before _1/_2 suffixes.
        files.sort(key=lambda p: (len(p.name), p.name))
        keep, drop = files[0], files[1:]
        print(f"\n  keep   {keep.name}")
        for d in drop:
            size = d.stat().st_size
            freed += size
            removed += 1
            print(f"  remove {d.name}  ({size // 1024} KB)")
            if args.apply:
                try:
                    d.unlink()
                except PermissionError:
                    print(f"         ! in use — stop the backend and rerun")

    total = sum(len(v) for v in by_hash.values())
    print(f"\n  {removed} duplicate(s) of {total} clips, {freed // (1024*1024)} MB")
    if not args.apply:
        print("  dry run — rerun with --apply to delete")


if __name__ == "__main__":
    main()
