"""Zero-shot (open-vocabulary) activity recognition with X-CLIP, measured.

X-CLIP scores a short video against text descriptions, so an activity the
Kinetics models were never trained on ("taking tablets") can be defined in
words. This measures, for a candidate model and prompt set:

  * on clips that DO show a new activity: how often it wins (recall);
  * on clips that do NOT: how often it wrongly wins (false alarms),
    especially on look-alikes (eating, drinking, cooking, lying down);
  * speed and GPU memory per look.

    python scripts/eval_open_vocab.py --model microsoft/xclip-base-patch32
    python scripts/eval_open_vocab.py --model microsoft/xclip-base-patch16-zero-shot --fp16

Clips: data/new-activities/ (local test clips, see its README) and the
labelled demo clips. Results: data/new-activities/results/<model>.json,
written after every clip so a restart loses nothing.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# Activity -> descriptions; their text embeddings are averaged.
PROMPTS: dict[str, list[str]] = {
    "brushing teeth": ["brushing teeth", "a person brushing their teeth with a toothbrush"],
    "cleaning": ["cleaning the floor", "a person mopping or cleaning"],
    "cooking": ["cooking food in a kitchen", "a person cooking on a stove"],
    "cycling": ["riding a bicycle", "a person cycling"],
    "dancing": ["dancing", "a person dancing"],
    "drinking": ["drinking from a cup or bottle", "a person drinking a beverage"],
    "eating": ["eating food", "a person eating a meal"],
    "exercising": ["exercising", "a person doing push ups, squats or pull ups"],
    "reading": ["reading a book", "a person reading"],
    "smoking": ["smoking a cigarette", "a person smoking"],
    "swimming": ["swimming", "a person swimming in a pool"],
    "using computer": ["typing on a computer keyboard", "a person using a laptop"],
    "using phone": ["using a mobile phone", "a person looking at a smartphone"],
    "writing": ["writing with a pen", "a person writing on paper"],
    "walking": ["walking", "a person walking"],
    "sitting": ["sitting", "a person sitting on a chair"],
    "lying down": ["lying down", "a person lying on a bed or sofa"],
    # new activities
    "making tea": ["making tea", "a person making a cup of tea, pouring hot water or boiling tea"],
    "taking tablets": ["taking a pill", "a person taking a tablet and swallowing medicine with water"],
    "falling down": ["falling down", "a person falling to the floor"],
}
NEW = ("making tea", "taking tablets", "falling down")


def test_clips() -> list[tuple[str, str | None]]:
    """(path, expected activity or None). Expected None = shows none of NEW."""
    d = ROOT / "data" / "new-activities"
    out: list[tuple[str, str | None]] = []
    named = {"mixkit-46792": "taking tablets", "mixkit-48203": "taking tablets",
             "mixkit-23814": "falling down", "commons-chai-kumbakonam": "making tea",
             "commons-tea-moroccan": "making tea", "commons-tea-traditional": "making tea"}
    for f in sorted(glob.glob(str(d / "*.*"))):
        stem = Path(f).stem
        if stem in named:
            out.append((f, named[stem]))
    for f in sorted(glob.glob(str(d / "urfall" / "urfall-*.mp4"))):
        out.append((f, "falling down" if "-fall-" in f else "no fall"))
    showcase = json.loads((ROOT / "data" / "showcase" / "showcase.json").read_text())
    for r in showcase["clips"]:
        if r.get("expected"):
            out.append((str(ROOT / "data" / "samples" / r["video"]), "other:" + r["expected"]))
    return out


def windows(path: str, nf: int, span_s: float = 2.5, step_s: float = 1.0) -> list[list[np.ndarray]]:
    """Sliding windows of nf frames spread over span_s, every step_s."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        s = 320 / min(h, w)
        frames.append(cv2.cvtColor(cv2.resize(f, (int(w * s), int(h * s))), cv2.COLOR_BGR2RGB))
    if not frames:
        return []
    span, step = max(nf, int(span_s * fps)), max(1, int(step_s * fps))
    starts = range(0, max(1, len(frames) - span + 1), step)
    return [[frames[min(len(frames) - 1, j)] for j in np.linspace(s, s + span - 1, nf).astype(int)]
            for s in starts][:20]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="microsoft/xclip-base-patch32")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from transformers import XCLIPModel, XCLIPProcessor

    proc = XCLIPProcessor.from_pretrained(args.model)
    dtype = torch.float16 if args.fp16 else torch.float32
    model = XCLIPModel.from_pretrained(args.model, torch_dtype=dtype).eval().to(args.device)
    nf = model.config.vision_config.num_frames
    names = list(PROMPTS)
    k = len(PROMPTS[names[0]])
    text = proc(text=[p for n in names for p in PROMPTS[n]], return_tensors="pt", padding=True)
    text = {kk: v.to(args.device) for kk, v in text.items()}

    out_dir = ROOT / "data" / "new-activities" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.model.split('/')[-1]}{'-fp16' if args.fp16 else ''}.json"
    results = json.loads(out.read_text()) if out.exists() else {}

    times = []
    for path, expected in test_clips():
        key = Path(path).name
        if key in results:
            continue
        wins = windows(path, nf)
        looks = []
        for clip in wins:
            v = proc(videos=[clip], return_tensors="pt")["pixel_values"].to(args.device, dtype)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(pixel_values=v, **text).logits_per_video[0].float()
            if args.device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            p = torch.softmax(logits.view(len(names), k).mean(1), -1).cpu().numpy()
            looks.append({"top": names[int(p.argmax())], **{n: round(float(p[names.index(n)]), 4) for n in NEW}})
        results[key] = {"expected": expected, "looks": looks}
        out.write_text(json.dumps({**results, "_meta": {
            "model": args.model, "fp16": args.fp16, "frames": nf,
            "ms_per_look": round(1000 * float(np.median(times)), 1) if times else None,
            "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if args.device == "cuda" else None,
        }}, indent=1))
        tops = [l["top"] for l in looks]
        print(f"{key:40} expected={expected!s:28} top: "
              + ", ".join(f"{t} {tops.count(t)}" for t in sorted(set(tops), key=tops.count, reverse=True)[:3]),
              flush=True)
    summarise(out)


def summarise(path: Path) -> None:
    res = json.loads(path.read_text())
    meta = res.pop("_meta", {})
    print(f"\n{meta.get('model')} fp16={meta.get('fp16')} {meta.get('frames')} frames: "
          f"{meta.get('ms_per_look')} ms/look, peak GPU {meta.get('peak_gpu_mb')} MB")
    for act in NEW:
        pos = [r for r in res.values() if r["expected"] == act]
        neg = [r for r in res.values() if r["expected"] != act]
        hit = sum(1 for r in pos if any(l["top"] == act for l in r["looks"]))
        share = np.mean([np.mean([l["top"] == act for l in r["looks"]]) for r in pos if r["looks"]]) if pos else 0
        fa_clips = [k for k, r in res.items() if r["expected"] != act
                    and np.mean([l["top"] == act for l in r["looks"]] or [0]) >= 0.25]
        fa_looks = np.mean([l["top"] == act for r in neg for l in r["looks"]]) if neg else 0
        print(f"  {act:15} found in {hit}/{len(pos)} clips (wins {share:.0%} of their looks); "
              f"false alarms: {fa_looks:.1%} of other looks, clips {fa_clips[:6]}")


if __name__ == "__main__":
    main()
