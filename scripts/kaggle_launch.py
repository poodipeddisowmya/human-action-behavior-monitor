"""Run the monitor on a Kaggle GPU notebook and share it with a public link.

Standard library only, so it runs on Kaggle's own Python before anything is
installed. Driven from kaggle/kaggle_demo.ipynb, one sub-command per cell:

    python scripts/kaggle_launch.py setup     # isolated env + models (~3-4 min, once)
    python scripts/kaggle_launch.py start     # server + public https link
    python scripts/kaggle_launch.py status    # health, FPS, GPU memory
    python scripts/kaggle_launch.py logs      # last lines of the server log
    python scripts/kaggle_launch.py stop

Design decisions, because each one fixes a way this fails on Kaggle:

* A separate virtual environment (uv, Python 3.11), not Kaggle's site-packages.
  The app pins numpy 1.26 / ultralytics 8.3.0; installing those over Kaggle's
  numpy 2 breaks pandas/scipy that ultralytics imports. The venv reproduces
  the exact versions this project was validated with, on any Kaggle image.
* Cloudflare quick tunnel for the public link: free, no account, HTTPS. HTTPS
  matters: browsers only grant webcam access on https, so "Go live" with the
  client's own camera works through it. ngrok is used instead when an
  NGROK_AUTHTOKEN secret is present.
* An access token, generated unless ACCESS_TOKEN is given, because the link
  is public. It is baked into the printed link; the page then sets a cookie.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
# Run from a notebook, this process inherits Jupyter's inline plotting
# backend, which the app's own environment does not have; every child
# process below inherits this instead (see config/settings.py).
os.environ["MPLBACKEND"] = "Agg"
# Off /kaggle/working on purpose: that directory is the notebook's saved
# output (20 GB cap, and everything in it is uploaded on "Save Version").
HOME = Path(os.environ.get("HAM_HOME", "/root/ham"))
VENV = HOME / "venv"
PY = VENV / "bin" / "python"
STATE = HOME / "state.json"
SERVER_LOG = HOME / "server.log"
TUNNEL_LOG = HOME / "tunnel.log"
PORT = int(os.environ.get("API_PORT", "8000"))
LOCAL = f"http://127.0.0.1:{PORT}"
CLOUDFLARED = HOME / "cloudflared"
CLOUDFLARED_URL = ("https://github.com/cloudflare/cloudflared/releases/latest/"
                   "download/cloudflared-linux-amd64")


def say(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str], check: bool = True, **kw) -> subprocess.CompletedProcess:
    say("  $ " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=check, **kw)


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def save_state(**kw) -> None:
    s = load_state()
    s.update(kw)
    HOME.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2))


def http_json(url: str, timeout: float = 3.0, token: str | None = None):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except Exception:
            return e.code, {}
    except Exception:
        return None, {}


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# --------------------------------------------------------------------- setup
def gpu_report() -> bool:
    smi = shutil.which("nvidia-smi")
    if not smi:
        say("\n!! No GPU in this session. In the notebook: Settings -> Accelerator -> "
            "GPU T4 x2, then restart. (The app still runs on CPU, slowly.)\n")
        return False
    out = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    say(f"GPU: {out.strip()}")
    return True


def internet_ok() -> bool:
    try:
        urllib.request.urlopen("https://pypi.org/simple/pip/", timeout=8)
        return True
    except Exception:
        say("\n!! No internet. In the notebook: Settings -> Internet -> On "
            "(needs a phone-verified Kaggle account).\n")
        return False


def setup() -> None:
    t0 = time.time()
    say("== 1/4  Checking the session")
    gpu_report()
    if not internet_ok():
        sys.exit(1)
    HOME.mkdir(parents=True, exist_ok=True)

    say("\n== 2/4  Python 3.11 environment (isolated from Kaggle's packages)")
    run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    uv = [sys.executable, "-m", "uv"]
    if not PY.exists():
        # uv fetches a managed 3.11 itself if this image ships another version.
        run(uv + ["venv", "--python", "3.11", "--seed", VENV])
    req_gpu = APP / "requirements-gpu.txt"
    pinned = run(uv + ["pip", "install", "--python", PY, "-r", req_gpu,
                       "--index-strategy", "unsafe-best-match"], check=False)
    if pinned.returncode != 0:
        say("  pinned GPU build unavailable; installing the current CUDA 12.6 build")
        run(uv + ["pip", "install", "--python", PY, "torch", "torchvision",
                  "--index-url", "https://download.pytorch.org/whl/cu126"])
        run(uv + ["pip", "install", "--python", PY, "-r", APP / "requirements.txt",
                  "--index-strategy", "unsafe-best-match"])

    say("\n== 3/4  Model weights")
    # CLIP (place names, sitting/standing). The app reads it offline-only.
    run([PY, "-c", "from transformers import CLIPModel, CLIPProcessor as P;"
                   "m='openai/clip-vit-base-patch32';CLIPModel.from_pretrained(m);"
                   "P.from_pretrained(m);print('  CLIP ready')"],
        env={**os.environ, "HF_HUB_OFFLINE": "0"})
    # VideoMAE, the GPU profile's activity model (config/model_config.yaml).
    run([PY, "-c", "from transformers import VideoMAEForVideoClassification as M,"
                   "VideoMAEImageProcessor as P;c='MCG-NJU/videomae-base-finetuned-kinetics';"
                   "M.from_pretrained(c);P.from_pretrained(c);print('  VideoMAE ready')"],
        env={**os.environ, "HF_HUB_OFFLINE": "0"})
    # SmolVLM2, the GPU profile's detailed-subtitle model (narrator.py).
    run([PY, "-c", "from transformers import AutoModelForImageTextToText as M,AutoProcessor as P;"
                   "c='HuggingFaceTB/SmolVLM2-500M-Video-Instruct';"
                   "M.from_pretrained(c);P.from_pretrained(c);print('  SmolVLM2 ready')"],
        env={**os.environ, "HF_HUB_OFFLINE": "0"})
    # X-CLIP, the open-vocabulary activities (making tea, taking tablets...).
    run([PY, "-c", "from transformers import XCLIPModel as M,XCLIPProcessor as P;"
                   "c='microsoft/xclip-base-patch16-zero-shot';"
                   "M.from_pretrained(c);P.from_pretrained(c);print('  X-CLIP ready')"],
        env={**os.environ, "HF_HUB_OFFLINE": "0"})
    # The GPU detector, if config asks for one larger than the bundled one.
    run([PY, "-c", "import sys;sys.path.insert(0,'.');from config import settings;"
                   "from src.tracking.person_tracker import fetch_weights;"
                   "print('  detector:', fetch_weights(settings.model.detector_weights,"
                   "settings.model.detector_fallback_weights))"], cwd=APP)

    say("\n== 4/4  GPU self-check")
    run([PY, "scripts/gpu_check.py"], cwd=APP, check=False)
    save_state(setup_done=True)
    say(f"Setup finished in {time.time() - t0:.0f}s. Next: start")


# --------------------------------------------------------------------- start
def start_server(token: str, source: str | None) -> int:
    st = load_state()
    if alive(st.get("server_pid")):
        say(f"server already running (pid {st['server_pid']})")
        return st["server_pid"]
    env = {**os.environ,
           "DEVICE": os.environ.get("DEVICE", "auto"),
           "ACCESS_TOKEN": token,
           "PROBE_CAMERAS": "0",                # no USB camera on Kaggle
           "API_PORT": str(PORT),
           "PYTHONUNBUFFERED": "1",
           "PYTHONIOENCODING": "utf-8"}
    cmd = [str(PY), "scripts/run_pipeline.py", "--host", "127.0.0.1", "--port", str(PORT)]
    if source:
        cmd += ["--source", source]
    log = SERVER_LOG.open("ab")
    proc = subprocess.Popen(cmd, cwd=APP, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)       # outlives this cell
    save_state(server_pid=proc.pid, token=token)
    say(f"server starting (pid {proc.pid}); loading models...")
    return proc.pid


def wait_ready(pid: int, token: str, timeout: float = 240) -> bool:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        if not alive(pid):
            say("\n!! The server exited during start-up. Last log lines:")
            tail(SERVER_LOG, 40)
            return False
        code, body = http_json(f"{LOCAL}/api/ready", token=token)
        if code == 200:
            say(f"server ready in {time.time() - t0:.0f}s")
            return True
        msg = body.get("reason") or ("loading models" if code is None else f"HTTP {code}")
        if msg != last:
            say(f"  ... {msg}")
            last = msg
        time.sleep(2)
    say("\n!! Not ready within the time limit. Last log lines:")
    tail(SERVER_LOG, 40)
    return False


def start_tunnel() -> str | None:
    if os.environ.get("NGROK_AUTHTOKEN"):
        return start_ngrok()
    if not CLOUDFLARED.exists():
        say("downloading cloudflared (public tunnel, no account needed)...")
        urllib.request.urlretrieve(CLOUDFLARED_URL, CLOUDFLARED)
        CLOUDFLARED.chmod(0o755)
    TUNNEL_LOG.write_text("")
    proc = subprocess.Popen(
        [str(CLOUDFLARED), "tunnel", "--no-autoupdate", "--url", LOCAL],
        stdout=TUNNEL_LOG.open("ab"), stderr=subprocess.STDOUT, start_new_session=True)
    save_state(tunnel_pid=proc.pid)
    pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    for _ in range(60):
        m = pat.search(TUNNEL_LOG.read_text(errors="ignore"))
        if m:
            return m.group(0)
        if proc.poll() is not None:
            break
        time.sleep(1)
    say("!! cloudflared did not produce a URL:")
    tail(TUNNEL_LOG, 20)
    return None


def start_ngrok() -> str | None:
    run([sys.executable, "-m", "pip", "install", "-q", "pyngrok"])
    from pyngrok import conf, ngrok           # noqa: E402  (installed just above)
    conf.get_default().auth_token = os.environ["NGROK_AUTHTOKEN"]
    return ngrok.connect(PORT, "http").public_url.replace("http://", "https://")


def start(source: str | None = None) -> None:
    if not PY.exists():
        say("Run setup first.")
        sys.exit(1)
    st = load_state()
    token = os.environ.get("ACCESS_TOKEN") or st.get("token") or secrets.token_urlsafe(12)
    pid = start_server(token, source)
    if not wait_ready(pid, token):
        sys.exit(1)
    url = None
    if not alive(st.get("tunnel_pid")) or not st.get("public_url"):
        url = start_tunnel()
        save_state(public_url=url)
    else:
        url = st["public_url"]
    code, m = http_json(f"{LOCAL}/api/metrics", token=token)
    say("\n" + "=" * 68)
    say(f"  Running on: {m.get('gpu') or m.get('device', '?')}")
    if url:
        say(f"  CLIENT LINK:  {url}/?token={token}")
        say("  (share this whole link; it works in any browser, phone included)")
    else:
        say("  No public link. The batch-processing cell below still works.")
    say(f"  Access token: {token}")
    say("=" * 68)
    say("Keep this notebook open during the demo: Kaggle stops idle sessions.")


# --------------------------------------------------------------------- ops
def tail(path: Path, n: int) -> None:
    try:
        lines = path.read_text(errors="ignore").splitlines()[-n:]
        say("\n".join(lines))
    except FileNotFoundError:
        say(f"(no {path.name} yet)")


def status() -> None:
    st = load_state()
    token = st.get("token")
    code, h = http_json(f"{LOCAL}/api/health")
    if code != 200:
        say("server: DOWN" + (" (process gone)" if not alive(st.get("server_pid")) else ""))
        return
    _, r = http_json(f"{LOCAL}/api/ready", token=token)
    _, m = http_json(f"{LOCAL}/api/metrics", token=token)
    gpu = f"{m.get('gpu')}  {m.get('gpu_mem_mb', 0)}/{m.get('gpu_mem_total_mb', 0)} MB" \
        if m.get("gpu") else m.get("device")
    say(f"server: {'READY' if r.get('ready') else 'NOT READY - ' + str(r.get('reason'))}"
        f" | {gpu} | {m.get('playback_fps', 0)} fps | detect {m.get('detect_ms', 0)} ms"
        f" | tracked {m.get('tracked', 0)} | viewers {m.get('viewers', 0)}"
        f" | restarts {m.get('restarts', 0)}")
    if st.get("public_url"):
        say(f"link: {st['public_url']}/?token={token}"
            f"  (tunnel {'up' if alive(st.get('tunnel_pid')) else 'DOWN - run start again'})")


def monitor(minutes: float) -> None:
    """Print status every minute; also keeps the Kaggle session active."""
    end = time.time() + minutes * 60
    while time.time() < end:
        status()
        st = load_state()
        if not alive(st.get("server_pid")):
            say("server died - restarting")
            start()
        time.sleep(60)


def stop() -> None:
    st = load_state()
    for key in ("tunnel_pid", "server_pid"):
        pid = st.get(key)
        if alive(pid):
            try:
                os.killpg(pid, signal.SIGTERM)
            except Exception:
                os.kill(pid, signal.SIGTERM)
            say(f"stopped {key.split('_')[0]} (pid {pid})")
    save_state(server_pid=None, tunnel_pid=None, public_url=None)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "setup":
        setup()
    elif cmd == "start":
        start(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "status":
        status()
    elif cmd == "monitor":
        monitor(float(sys.argv[2]) if len(sys.argv) > 2 else 120)
    elif cmd == "logs":
        tail(SERVER_LOG, int(sys.argv[2]) if len(sys.argv) > 2 else 60)
    elif cmd == "stop":
        stop()
    else:
        say(__doc__)


if __name__ == "__main__":
    main()
