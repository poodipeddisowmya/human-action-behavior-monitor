"""Unattended end-to-end validation on a Kaggle GPU (a "script" kernel).

Pushed and run by scripts/kaggle_auto.py; it can also be pasted into any
Kaggle notebook with the project dataset attached. Every stage is recorded
in /kaggle/working/gpu_report.json and gpu_report.md (downloaded as the
kernel's output), and a failing stage never stops the stages after it.

Stages:
    env        GPU, driver, Python, internet
    setup      isolated env + weights (scripts/kaggle_launch.py setup)
    gpu_check  per-model latency on this GPU
    accuracy   15-clip demo set on the GPU, shipped detector vs. bundled nano
    server     the real server + public tunnel
    smoke      real-time checks locally AND through the public tunnel
    batch      an annotated report video
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

OUT = Path("/kaggle/working")
APP = Path("/root/app")             # not /kaggle/working: keep the output small
VENV_PY = "/root/ham/venv/bin/python"
REPORT: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "stages": {}}
LOGS = OUT / "logs"


def record(stage: str, ok: bool, secs: float, **info) -> None:
    REPORT["stages"][stage] = {"ok": ok, "seconds": round(secs, 1), **info}
    (OUT / "gpu_report.json").write_text(json.dumps(REPORT, indent=2, default=str))
    print(f"\n##### {stage}: {'OK' if ok else 'FAILED'} ({secs:.0f}s) {info if not ok else ''}\n",
          flush=True)


def sh(cmd, log: str, env=None, timeout=3600) -> tuple[int, str]:
    LOGS.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(cmd, cwd=APP, env={**os.environ, **(env or {})}, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       encoding="utf-8", errors="replace", shell=isinstance(cmd, str))
    (LOGS / f"{log}.log").write_text(p.stdout)
    print(p.stdout[-4000:], flush=True)
    return p.returncode, p.stdout


def stage(name):
    def wrap(fn):
        def run():
            t0 = time.time()
            try:
                ok, info = fn()
            except Exception as e:
                import traceback
                ok, info = False, {"error": repr(e), "trace": traceback.format_exc()[-1500:]}
            record(name, ok, time.time() - t0, **info)
            return ok
        return run
    return wrap


def _gpu() -> str | None:
    smi = shutil.which("nvidia-smi") or next(
        (p for p in ("/usr/bin/nvidia-smi", "/usr/local/nvidia/bin/nvidia-smi") if Path(p).exists()),
        None)
    if smi:
        r = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    try:                                   # Kaggle's own torch, if nvidia-smi is hidden
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


@stage("env")
def env():
    gpu = _gpu()
    try:
        import urllib.request
        urllib.request.urlopen("https://pypi.org/simple/pip/", timeout=10)
        net = True
    except Exception:
        net = False
    inputs = sorted(str(p) for p in Path("/kaggle/input").rglob("*") if p.is_dir())[:15] \
        if Path("/kaggle/input").exists() else []
    # Locate the project: an extracted dataset folder, or a zip inside one.
    if not (APP / "scripts" / "run_pipeline.py").exists():
        found = glob.glob("/kaggle/input/**/scripts/run_pipeline.py", recursive=True)
        if found:
            shutil.copytree(Path(found[0]).parent.parent, APP)
        else:
            zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
            if not zips:
                return False, {"error": "project dataset not attached", "inputs": inputs,
                               "gpu": gpu}
            with zipfile.ZipFile(zips[0]) as z:
                z.extractall("/root/unzipped")
            root = next(Path("/root/unzipped").rglob("run_pipeline.py")).parent.parent
            shutil.move(str(root), APP)
    info = {"gpu": gpu, "internet": net, "python": sys.version.split()[0],
            "clips": len(list((APP / "data" / "samples").glob("*.*"))), "inputs": inputs}
    if not gpu:
        info["error"] = ("no GPU in this session: the account needs phone verification "
                         "for GPUs (kaggle.com -> Settings), or the T4 queue was full")
    return bool(gpu and net), info


@stage("setup")
def setup():
    code, out = sh([sys.executable, "scripts/kaggle_launch.py", "setup"], "setup")
    return code == 0, {"tail": out[-600:]} if code else {}


@stage("gpu_check")
def gpu_check():
    code, out = sh([VENV_PY, "scripts/gpu_check.py"], "gpu_check")
    timings = dict(re.findall(r"^\s+(.+?)\s+:\s+([\d.]+) ms", out, re.M))
    uses = re.search(r"App will use: (\S+)", out)
    return code == 0 and bool(uses) and uses.group(1).startswith("cuda"), \
        {"device": uses.group(1) if uses else None, "ms": timings}


def _eval(weights: str | None) -> dict:
    env = {"PYTHONIOENCODING": "utf-8"}
    if weights:
        env["DETECTOR_WEIGHTS"] = weights
    # --realtime: score what a viewer sees (clips at native speed), see eval_offline.
    code, out = sh([VENV_PY, "scripts/eval_offline.py", "--realtime"],
                   f"eval_{weights or 'shipped'}", env=env)
    rows = re.findall(r"^\s+(\S+)\s+(.+?)\s{2,}(.+?)\s+(PASS|FAIL)\s+([\d.]+)$", out, re.M)
    score = re.search(r"(\d+)/(\d+) =", out)
    return {"score": score.group(0).rstrip(" =") if score else None,
            "passed": int(score.group(1)) if score else 0,
            "total": int(score.group(2)) if score else 0,
            "per_frame_fps_median": sorted(float(r[4]) for r in rows)[len(rows) // 2] if rows else None,
            "failures": [f"{r[0]}: expected {r[1].strip()}, got {r[2].strip()}"
                         for r in rows if r[3] == "FAIL"]}


@stage("accuracy")
def accuracy():
    shipped = _eval(None)
    nano = _eval("yolov8n-pose.pt")
    ok = shipped["total"] > 0 and shipped["passed"] >= nano["passed"]
    return ok, {"shipped_gpu_detector": shipped, "bundled_nano": nano}


STATE = Path("/root/ham/state.json")


@stage("server")
def server():
    code, out = sh([sys.executable, "scripts/kaggle_launch.py", "start"], "start",
                   env={"ACCESS_TOKEN": "validation-" + os.urandom(6).hex()})
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    return code == 0, {"public_url": st.get("public_url"), "tail": out[-400:] if code else ""}


def _smoke(base: str, token: str, name: str, min_stream: float) -> tuple[bool, dict]:
    code, out = sh([VENV_PY, "scripts/smoke_test.py", "--base", base, "--token", token,
                    "--json", str(OUT / f"smoke_{name}.json"),
                    "--min-stream-fps", str(min_stream)], f"smoke_{name}")
    data = json.loads((OUT / f"smoke_{name}.json").read_text()) \
        if (OUT / f"smoke_{name}.json").exists() else {}
    return code == 0, {c["check"]: {k: v for k, v in c.items() if k != "check"}
                       for c in data.get("checks", [])}


@stage("smoke_local")
def smoke_local():
    st = json.loads(STATE.read_text())
    return _smoke("http://127.0.0.1:8000", st["token"], "local", 15)


@stage("live_accuracy")
def live_accuracy():
    """What the dashboard displays, clip by clip — the number that matters."""
    st = json.loads(STATE.read_text())
    code, out = sh([VENV_PY, "scripts/live_accuracy.py", "--base", "http://127.0.0.1:8000",
                    "--token", st["token"], "--json", str(OUT / "live_accuracy.json")],
                   "live_accuracy")
    data = json.loads((OUT / "live_accuracy.json").read_text())
    return code == 0, {"score": f"{data['passed']}/{data['total']}",
                       "clips": {r["clip"]: f"{r['result']} {r['got']} "
                                 f"(correct after {r['correct_after_s']} s)"
                                 for r in data["clips"]}}


@stage("smoke_public")
def smoke_public():
    st = json.loads(STATE.read_text())
    if not st.get("public_url"):
        return False, {"error": "no public tunnel URL"}
    return _smoke(st["public_url"], st["token"], "public", 8)


@stage("batch")
def batch():
    code, out = sh([VENV_PY, "scripts/process_video.py", "data/samples/demo-cooking.mp4",
                    "data/samples/office-people.mp4", "--out", str(OUT / "reports")], "batch")
    sums = [json.loads(p.read_text()) for p in (OUT / "reports").glob("*_summary.json")]
    return code == 0 and len(sums) == 2, {
        "videos": [{k: s[k] for k in ("video", "processing_fps", "people", "events", "by_label")}
                   for s in sums]}


def write_markdown() -> None:
    lines = [f"# GPU validation — {REPORT['started']}", ""]
    for name, s in REPORT["stages"].items():
        lines.append(f"## {'✅' if s['ok'] else '❌'} {name}  ({s['seconds']}s)")
        lines.append("```json")
        lines.append(json.dumps({k: v for k, v in s.items() if k not in ("ok", "seconds")},
                                indent=2, default=str, ensure_ascii=False))
        lines.append("```")
    (OUT / "gpu_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    # Reproduce the interactive notebook's environment, not a bare script's:
    # Jupyter sets this for every child process, and it once crashed the app
    # on import in the live demo notebook while every script-kernel run passed.
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    ran = {}
    ran["env"] = env()
    if ran["env"]:                          # no GPU / no internet: nothing worth testing
        ran["setup"] = setup()
    if ran.get("setup"):
        gpu_check()
        accuracy()
        if server():
            smoke_local()
            live_accuracy()
            smoke_public()
        batch()
        subprocess.run([sys.executable, "scripts/kaggle_launch.py", "stop"], cwd=APP)
        shutil.copy("/root/ham/server.log", LOGS / "server.log") \
            if Path("/root/ham/server.log").exists() else None
    REPORT["finished"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    REPORT["all_ok"] = all(s["ok"] for s in REPORT["stages"].values())
    (OUT / "gpu_report.json").write_text(json.dumps(REPORT, indent=2, default=str))
    write_markdown()
    print("\n" + (OUT / "gpu_report.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
