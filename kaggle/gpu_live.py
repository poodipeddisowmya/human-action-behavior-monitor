"""Host the live dashboard on a Kaggle GPU as a background job.

Pushed by `scripts/kaggle_auto.py live --hours N`, which reads the client
link from this job's live log. Runs setup, starts the server and the public
tunnel, prints the link, then keeps the server healthy until either
HOURS have passed or nobody has watched for IDLE_MIN minutes, so a
forgotten demo does not burn the weekly GPU quota.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HOURS = 3.0                 # replaced by kaggle_auto.py at push time
IDLE_MIN = 45               # stop after this long with no viewer
APP = Path("/kaggle/working/human-action-monitor")
STATE = Path("/root/ham/state.json")
ENV = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}


def launch(*args: str) -> int:
    return subprocess.run([sys.executable, "scripts/kaggle_launch.py", *args],
                          cwd=APP, env=ENV).returncode


def metrics(token: str) -> dict:
    req = urllib.request.Request("http://127.0.0.1:8000/api/metrics",
                                 headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def main() -> None:
    if not (APP / "scripts" / "run_pipeline.py").exists():
        src = Path(glob.glob("/kaggle/input/**/scripts/run_pipeline.py", recursive=True)[0])
        shutil.copytree(src.parent.parent, APP)
    if launch("setup") != 0 or launch("start") != 0:
        launch("logs", "80")
        sys.exit("live demo failed to start (log above)")

    st = json.loads(STATE.read_text())
    link = f"{st.get('public_url')}/?token={st.get('token')}"
    print(f"\nCLIENT LINK: {link}\n", flush=True)

    t_end = time.time() + HOURS * 3600
    last_seen = time.time()
    while time.time() < t_end:
        time.sleep(60)
        m = metrics(st["token"])
        if not m:
            print("server not answering - restarting", flush=True)
            launch("start")
            st = json.loads(STATE.read_text())
            print(f"CLIENT LINK: {st.get('public_url')}/?token={st.get('token')}", flush=True)
            continue
        if m.get("viewers", 0) or m.get("browser_camera_clients", 0):
            last_seen = time.time()
        idle = (time.time() - last_seen) / 60
        print(f"[{time.strftime('%H:%M')}] {m.get('gpu')} | {m.get('playback_fps')} fps | "
              f"viewers {m.get('viewers')} | idle {idle:.0f} min", flush=True)
        if idle >= IDLE_MIN:
            print(f"no viewer for {IDLE_MIN} min - shutting down to save GPU quota", flush=True)
            break
    launch("stop")
    print("live demo stopped", flush=True)


if __name__ == "__main__":
    main()
