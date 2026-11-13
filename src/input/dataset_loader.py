"""Helpers for loading small, task-relevant slices of public action datasets.

We deliberately do NOT bulk-download entire datasets. Selection follows the
'take only what you need' principle from the master brief.

Recommended datasets:
  - UCF101 / HMDB51 : good coverage of walk/run/sit/stand actions.
  - NTU RGB+D subset: skeleton ground-truth for validation of ST-GCN.
  - Le2i Fall Detection : realistic falls for the safety class.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetClip:
    path: Path
    label: str


def list_clips(root: Path, labels: list[str] | None = None) -> list[DatasetClip]:
    """List clips under `root/<label>/*.mp4|avi`.

    Directory convention:  root/standing/vid_001.mp4
    """
    root = Path(root)
    if not root.exists():
        return []
    clips: list[DatasetClip] = []
    for label_dir in root.iterdir():
        if not label_dir.is_dir():
            continue
        if labels and label_dir.name not in labels:
            continue
        for f in label_dir.glob("*.*"):
            if f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                clips.append(DatasetClip(path=f, label=label_dir.name))
    return clips
