"""Do the whole Kaggle round-trip from this PC, unattended.

    .venv\\Scripts\\python.exe scripts\\kaggle_auto.py validate   # test on a real T4, get a report
    .venv\\Scripts\\python.exe scripts\\kaggle_auto.py demo       # put the live-demo notebook in your account
    .venv\\Scripts\\python.exe scripts\\kaggle_auto.py all

validate:
    1. builds dist/human-action-monitor.zip
    2. uploads it as a PRIVATE Kaggle dataset (new version if it exists)
    3. runs kaggle/gpu_validation.py on a Kaggle T4 (GPU + internet on)
    4. waits, downloads the results to dist/kaggle-output/, prints the report
demo:
    saves kaggle/kaggle_demo.ipynb as a private notebook with the dataset,
    GPU and internet already attached, without running it (no quota used).
    Open it on kaggle.com -> Edit -> Run All to get the client link.

Sign in once:  .venv\\Scripts\\kaggle.exe auth login   (browser), or save an
API token from kaggle.com -> Settings -> API as %USERPROFILE%\\.kaggle\\access_token
Internet in GPU sessions also needs a phone-verified Kaggle account.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
DATASET_SLUG = "human-action-monitor"
VALIDATE_SLUG = "ham-gpu-validation"
DEMO_SLUG = "ham-gpu-demo"


def kaggle(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    # KAGGLE_CLI, or the one beside this Python, or any on PATH. The Jetson's
    # Python 3.10 only gets kaggle 1.7, which cannot use the new KGAT_ tokens;
    # there the current CLI runs from its own environment (uv tool install
    # --python 3.12 kaggle --with six), found on PATH.
    exe = (os.environ.get("KAGGLE_CLI")
           or shutil.which("kaggle", path=str(Path(sys.executable).parent))
           or shutil.which("kaggle"))
    cmd = [exe, *args] if exe else [sys.executable, "-m", "kaggle", *args]
    print("  $ kaggle " + " ".join(args), flush=True)
    return subprocess.run(cmd, check=check, text=True,
                          capture_output=capture, encoding="utf-8", errors="replace")


def ensure_cli() -> None:
    if os.environ.get("KAGGLE_CLI") or shutil.which("kaggle"):
        return
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("installing the Kaggle CLI...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
    except Exception:
        pass        # imported but could not authenticate; reported below


def username(cli_user: str | None) -> str:
    """The signed-in Kaggle account, whichever way the CLI authenticated."""
    if cli_user:
        return cli_user
    out = kaggle("config", "view", check=False, capture=True)
    m = re.search(r"username:\s*(\S+)", out.stdout or "")
    if m and m.group(1).lower() != "none":
        return m.group(1)
    sys.exit("\nNot signed in to Kaggle. Do ONE of these, then run this again:\n"
             "  a) .venv\\Scripts\\kaggle.exe auth login        (opens the browser)\n"
             "  b) kaggle.com -> Settings -> API -> Generate New Token, and save the\n"
             f"     token text as {Path.home() / '.kaggle' / 'access_token'}\n")


# --------------------------------------------------------------------- dataset
def upload_dataset(user: str, wait_s: int = 3600) -> str:
    ref = f"{user}/{DATASET_SLUG}"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "package_for_kaggle.py")], check=True)
    stage = DIST / "kaggle-dataset"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    shutil.copy(DIST / "human-action-monitor.zip", stage / "human-action-monitor.zip")
    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": "Human Action Monitor (app bundle)", "id": ref,
        "licenses": [{"name": "other"}]}, indent=2))

    exists = kaggle("datasets", "status", ref, check=False, capture=True).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(stage), "-m",
               time.strftime("update %Y-%m-%d %H:%M"), "-q")
    else:
        kaggle("datasets", "create", "-p", str(stage), "-q")     # private by default
    # Wait for THIS build to be live, not just for "ready": straight after a
    # new version is submitted, Kaggle still reports the previous version as
    # ready, and a GPU job started then silently gets the old files (it
    # happened: a benchmark ran against yesterday's code).
    marker = f"BUILD-{(DIST / 'build_id.txt').read_text().strip()}.txt"
    print(f"waiting for Kaggle to publish this build ({marker})...", flush=True)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        st = kaggle("datasets", "status", ref, check=False, capture=True).stdout.strip().lower()
        if "error" in st:
            sys.exit(f"dataset processing failed: {st}")
        if "ready" in st:
            files = kaggle("datasets", "files", ref, "--page-size", "200",
                           check=False, capture=True).stdout
            if marker in files:
                print(f"dataset ready: https://www.kaggle.com/datasets/{ref}", flush=True)
                time.sleep(30)          # let the new version reach the kernel workers
                return ref
        time.sleep(20)
    sys.exit("this build is still not live on Kaggle; run again in a few minutes")


# --------------------------------------------------------------------- kernels
def push_kernel(user: str, slug: str, title: str, code: Path, ktype: str, dataset: str,
                run: bool = True) -> str:
    ref = f"{user}/{slug}"
    stage = DIST / f"kaggle-{slug}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    shutil.copy(code, stage / code.name)
    meta = {"id": ref, "title": title, "code_file": code.name, "language": "python",
            "kernel_type": ktype, "is_private": True, "enable_gpu": True,
            "enable_internet": True, "dataset_sources": [dataset],
            "competition_sources": [], "kernel_sources": [],
            "machine_shape": "NvidiaTeslaT4"}
    (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    args = ["kernels", "push", "-p", str(stage), "--accelerator", "NvidiaTeslaT4"]
    # --no-run: save the notebook without executing it (no GPU quota used).
    args += ["-t", "5400"] if run else ["--no-run"]
    r = kaggle(*args, check=False, capture=True)
    print((r.stdout or "") + (r.stderr or ""), flush=True)
    if r.returncode != 0 or "error" in ((r.stdout or "") + (r.stderr or "")).lower():
        sys.exit("kernel push failed (see above). If it mentions phone verification, "
                 "verify your phone at kaggle.com -> Settings, then run again.")
    return ref


def wait_kernel(ref: str, timeout_min: int = 120) -> str:
    print(f"running on Kaggle: https://www.kaggle.com/code/{ref}  (checking every minute)",
          flush=True)
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout_min * 60:
        out = kaggle("kernels", "status", ref, check=False, capture=True).stdout.lower()
        for state in ("complete", "error", "cancel"):
            if state in out:
                print(f"kernel finished: {state} after {(time.time() - t0) / 60:.0f} min")
                return state
        cur = "running" if "running" in out else "queued" if "queued" in out else out.strip()
        if cur != last:
            print(f"  ... {cur}", flush=True)
            last = cur
        time.sleep(60)
    return "timeout"


def validate(user: str, dataset: str) -> int:
    ref = push_kernel(user, VALIDATE_SLUG, "HAM GPU validation",
                      ROOT / "kaggle" / "gpu_validation.py", "script", dataset)
    time.sleep(30)                        # the push takes a moment to register
    state = wait_kernel(ref)
    out = DIST / "kaggle-output"
    shutil.rmtree(out, ignore_errors=True)
    kaggle("kernels", "output", ref, "-p", str(out), check=False)
    report = out / "gpu_report.md"
    if report.exists():
        print("\n" + report.read_text(encoding="utf-8"))
        data = json.loads((out / "gpu_report.json").read_text())
        print(f"\nRESULT: {'ALL STAGES PASSED' if data.get('all_ok') else 'SOME STAGES FAILED'}"
              f"   (full output in {out})")
        return 0 if data.get("all_ok") else 1
    print(f"no report produced (kernel state: {state}). Log: https://www.kaggle.com/code/{ref}")
    return 1


def experiment(user: str, dataset: str) -> int:
    """kaggle/gpu_experiment.py: which GPU setting changes the labels."""
    ref = push_kernel(user, "ham-gpu-experiment", "HAM GPU Experiment",
                      ROOT / "kaggle" / "gpu_experiment.py", "script", dataset)
    time.sleep(30)
    wait_kernel(ref)
    out = DIST / "kaggle-experiment"
    shutil.rmtree(out, ignore_errors=True)
    kaggle("kernels", "output", ref, "-p", str(out), check=False)
    res = out / "experiment.json"
    if not res.exists():
        print(f"no result; log: https://www.kaggle.com/code/{ref}")
        return 1
    print(res.read_text())
    return 0


def live(user: str, dataset: str, hours: float) -> int:
    """Host the live dashboard on a T4 and print the client link."""
    src = (ROOT / "kaggle" / "gpu_live.py").read_text(encoding="utf-8")
    code = DIST / "gpu_live.py"
    DIST.mkdir(exist_ok=True)
    code.write_text(src.replace("HOURS = 3.0 ", f"HOURS = {hours} "), encoding="utf-8")
    ref = f"{user}/ham-gpu-live"
    stage = DIST / "kaggle-ham-gpu-live"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    shutil.copy(code, stage / "gpu_live.py")
    (stage / "kernel-metadata.json").write_text(json.dumps({
        "id": ref, "title": "HAM GPU Live", "code_file": "gpu_live.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "dataset_sources": [dataset], "competition_sources": [],
        "kernel_sources": [], "machine_shape": "NvidiaTeslaT4"}, indent=2))
    r = kaggle("kernels", "push", "-p", str(stage), "--accelerator", "NvidiaTeslaT4",
               "-t", str(int(hours * 3600 + 900)), check=False, capture=True)
    print((r.stdout or "") + (r.stderr or ""), flush=True)
    if r.returncode != 0:
        return 1
    print("starting the GPU session; the link appears in about 4-6 minutes...", flush=True)
    t0 = time.time()
    while time.time() - t0 < 20 * 60:
        exe = shutil.which("kaggle", path=str(Path(sys.executable).parent))
        p = subprocess.Popen([exe, "kernels", "logs", ref, "-f"], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace")
        try:
            for line in p.stdout:
                if "CLIENT LINK:" in line:
                    link = line.split("CLIENT LINK:", 1)[1].strip()
                    print(f"\n{'=' * 70}\n  LIVE ON GPU:  {link}\n"
                          f"  Runs up to {hours:g} h; stops itself after 45 min with no viewer.\n"
                          f"  Stop early: https://www.kaggle.com/code/{ref} -> Cancel run\n"
                          f"{'=' * 70}", flush=True)
                    (DIST / "live_link.txt").write_text(link)
                    return 0
                if "failed to start" in line:
                    print(line, flush=True)
                    return 1
        finally:
            p.kill()
        time.sleep(20)                    # session not up yet: try again
    print(f"no link yet; watch https://www.kaggle.com/code/{ref}")
    return 1


def benchmark(user: str, dataset: str) -> int:
    """kaggle/gpu_benchmark.py: every candidate improvement, measured on a T4."""
    ref = push_kernel(user, "ham-gpu-benchmark", "HAM GPU Benchmark",
                      ROOT / "kaggle" / "gpu_benchmark.py", "script", dataset)
    time.sleep(30)
    wait_kernel(ref, timeout_min=180)
    out = DIST / "kaggle-benchmark"
    shutil.rmtree(out, ignore_errors=True)
    kaggle("kernels", "output", ref, "-p", str(out), check=False)
    found = list(out.rglob("benchmark.json"))
    if not found:
        print(f"no benchmark produced; log: https://www.kaggle.com/code/{ref}")
        return 1
    data = json.loads(found[0].read_text())
    print("probe:", json.dumps(data.get("probe"), indent=1))
    for name, cfg in data["configs"].items():
        print(f"{name:<16} {json.dumps(cfg['summary'])}")
    return 0


def showcase(user: str, dataset: str) -> int:
    """Record every demo clip on the GPU; build dist/client-demo/index.html."""
    ref = push_kernel(user, "ham-gpu-showcase", "HAM GPU Showcase",
                      ROOT / "kaggle" / "gpu_showcase.py", "script", dataset)
    time.sleep(30)
    wait_kernel(ref)
    raw = DIST / "kaggle-showcase"
    shutil.rmtree(raw, ignore_errors=True)
    kaggle("kernels", "output", ref, "-p", str(raw), check=False)
    found = list(raw.rglob("showcase.json"))
    if not found:
        print(f"no showcase produced; log: https://www.kaggle.com/code/{ref}")
        return 1
    out = DIST / "client-demo"
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(found[0].parent, out)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_comparison.py"), str(out)],
                   check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_showcase.py"), str(out)],
                   check=True)
    print(f"\nClient demo ready: {out / 'index.html'}")
    return 0


def demo(user: str, dataset: str) -> None:
    # Kaggle derives the URL from the title, so the two must match.
    ref = push_kernel(user, DEMO_SLUG, "HAM GPU Demo",
                      ROOT / "kaggle" / "kaggle_demo.ipynb", "notebook", dataset, run=False)
    print(f"\nDemo notebook ready: https://www.kaggle.com/code/{ref}\n"
          "Open it -> Edit -> press Run All. The client link appears in about 4 minutes.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["validate", "demo", "all", "experiment", "showcase", "benchmark", "live"])
    ap.add_argument("--user", help="Kaggle username (default: from kaggle.json)")
    ap.add_argument("--hours", type=float, default=3.0,
                    help="live: keep the demo up at most this long")
    ap.add_argument("--skip-upload", action="store_true",
                    help="reuse the dataset already on Kaggle")
    args = ap.parse_args()

    ensure_cli()
    user = username(args.user)
    dataset = f"{user}/{DATASET_SLUG}" if args.skip_upload else upload_dataset(user)
    code = 0
    if args.action == "experiment":
        sys.exit(experiment(user, dataset))
    if args.action == "showcase":
        sys.exit(showcase(user, dataset))
    if args.action == "benchmark":
        sys.exit(benchmark(user, dataset))
    if args.action == "live":
        sys.exit(live(user, dataset, args.hours))
    if args.action in ("validate", "all"):
        code = validate(user, dataset)
    if args.action in ("demo", "all"):
        demo(user, dataset)
    sys.exit(code)


if __name__ == "__main__":
    main()
