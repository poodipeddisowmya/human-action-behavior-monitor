"""Accuracy as the viewer sees it: score the labels shown by a running server.

    python scripts/live_accuracy.py --base http://127.0.0.1:8000 --token XYZ

For each labelled demo clip: switch the live server to it, sample the labels
on screen (/api/stream/live) for --seconds, and apply exactly the same
verdict rule as eval_offline.py. This is the number that matters for a demo
or a deployment: warm models, real-time playback, looping clips — what the
dashboard actually displays. Also reports how long the correct label took
to appear after the switch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_offline import EXPECTED, verdict  # noqa: E402

# Labels from the first moments after a switch belong to the transition.
_SETTLE_S = 1.0


def score_clip(c: httpx.Client, clip: str, seconds: float) -> dict:
    c.post("/api/sources", json={"filename": clip}).raise_for_status()
    t0 = time.time()
    labels: Counter = Counter()
    first_ok = None
    expected = EXPECTED[clip]
    while time.time() - t0 < seconds:
        live = c.get("/api/stream/live").json()
        now = time.time() - t0
        for p in live.get("people", []):
            if now >= _SETTLE_S:
                labels[p["label"]] += 1
            if first_ok is None and p["label"] == expected:
                first_ok = round(now, 1)
        time.sleep(0.25)
    mark, got = verdict(expected, labels)
    return {"clip": clip, "expected": expected, "got": got, "result": mark,
            "correct_after_s": first_ok, "labels": dict(labels.most_common(4))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--token", default="")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    H = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    samples = Path(__file__).resolve().parent.parent / "data" / "samples"

    rows = []
    with httpx.Client(base_url=args.base.rstrip("/"), headers=H, timeout=15) as c:
        device = c.get("/api/health").json().get("gpu") or "cpu"
        print(f"\n  live accuracy on {device}, {args.seconds:.0f} s per clip\n")
        print(f"  {'clip':<30} {'expected':<15} {'got':<16} {'':<4} {'correct after':>13}")
        print("  " + "-" * 82)
        for clip in EXPECTED:
            if not (samples / clip).exists():
                continue
            r = score_clip(c, clip, args.seconds)
            rows.append(r)
            after = f"{r['correct_after_s']} s" if r["correct_after_s"] is not None else "-"
            print(f"  {clip[:30]:<30} {str(r['expected']):<15} {str(r['got'])[:16]:<16} "
                  f"{r['result']:<4} {after:>13}", flush=True)
    passed = sum(r["result"] == "PASS" for r in rows)
    print(f"\n  {passed}/{len(rows)} = {passed / max(1, len(rows)):.0%}\n")
    if args.json:
        args.json.write_text(json.dumps({"device": device, "passed": passed,
                                         "total": len(rows), "clips": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
