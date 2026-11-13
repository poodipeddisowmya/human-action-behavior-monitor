"""Install a CPU-vs-GPU recording into the dashboard's "CPU vs GPU" tab.

    python scripts/install_showcase.py dist/client-demo

Copies showcase.json and every video recorded on BOTH sides (GPU side from
the folder, CPU side from its before/ folder) into data/showcase/, which
/api/showcase serves. Only clips present on both sides are copied, so the
tab never shows a one-sided comparison.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils.showcase import summarize  # noqa: E402


def main(src: Path) -> None:
    data = json.loads((src / "showcase.json").read_text(encoding="utf-8"))
    dst = ROOT / "data" / "showcase"
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "before").mkdir(parents=True)
    pairs = summarize(data)["clips"]
    for p in pairs:
        shutil.copy(src / p["right"]["video"], dst / p["right"]["video"])
        shutil.copy(src / "before" / p["left"]["video"], dst / "before" / p["left"]["video"])
    (dst / "showcase.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    size = sum(f.stat().st_size for f in dst.rglob("*.mp4")) / 2**20
    print(f"  installed {len(pairs)} clip pairs ({size:.0f} MB) into {dst}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "dist/client-demo"))
