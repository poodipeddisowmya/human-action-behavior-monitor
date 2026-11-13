"""Launch the API + pipeline together.

Usage:
    python scripts/run_pipeline.py                       # first bundled demo clip
    python scripts/run_pipeline.py --source data/samples/demo-eating.avi
    python scripts/run_pipeline.py --source 0            # webcam

Then open http://localhost:8000 — the built UI is served from the same port,
and any clip can be selected or uploaded from there.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

from config import settings
from src.api.main import app, set_pipeline_source
from src.input import CameraSource, SwappableFileSource


def build_source_factory(target: str):
    if target.isdigit():
        return lambda: CameraSource(int(target))
    # Files are wrapped in SwappableFileSource so the UI can switch mid-run.
    return lambda: SwappableFileSource(target)


def default_source() -> str:
    """First bundled demo clip, so the app starts with no arguments.

    Requiring --source meant a new user's first command was an error. The
    source is switchable from the UI anyway, so the initial one only needs to
    be something valid.
    """
    samples = settings.data_dir / "samples"
    # A clip the client recognises makes a better first impression than
    # whatever sorts first alphabetically.
    preferred = samples / "demo-cooking.mp4"
    if preferred.exists():
        return str(preferred)
    samples = sorted(samples.glob("*.avi")) + sorted(samples.glob("*.mp4"))
    if not samples:
        raise SystemExit(
            f"No clips found in {settings.data_dir / 'samples'}.\n"
            f"Pass one explicitly:  python scripts/run_pipeline.py --source <video>"
        )
    return str(samples[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None,
                        help="video file or webcam index (e.g. '0'); "
                             "defaults to the first bundled demo clip")
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    args = parser.parse_args()

    source = args.source or default_source()
    set_pipeline_source(build_source_factory(source))

    url = f"http://localhost:{args.port}"
    print(f"\n  Starting on {url}   (source: {Path(source).name}, device: {settings.device})")
    print(f"  Open {url} in a browser. Ctrl+C to stop.\n", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                # Behind a tunnel / reverse proxy: trust X-Forwarded-* so the
                # app knows the browser is on https (secure cookies, wss).
                proxy_headers=True, forwarded_allow_ips="*",
                # MJPEG viewers hold a connection open forever; don't let
                # shutdown wait on them indefinitely.
                timeout_graceful_shutdown=5)


if __name__ == "__main__":
    main()
