"""Objective evaluation: accuracy, precision/recall/F1, FPS, RAM, GPU."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Iterable

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class ActionEval:
    accuracy: float
    precision: float
    recall: float
    f1: float
    report: str
    confusion: list[list[int]]


def evaluate_actions(y_true: Iterable[str], y_pred: Iterable[str], labels: list[str]) -> ActionEval:
    y_true, y_pred = list(y_true), list(y_pred)
    return ActionEval(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        recall=recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        f1=f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        report=classification_report(y_true, y_pred, labels=labels, zero_division=0),
        confusion=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    )


@dataclass
class PerformanceMonitor:
    frames: int = 0
    started: float = field(default_factory=time.time)

    def tick(self) -> None:
        self.frames += 1

    @property
    def fps(self) -> float:
        elapsed = time.time() - self.started
        return self.frames / elapsed if elapsed else 0.0

    def snapshot(self) -> dict:
        import psutil
        p = psutil.Process(os.getpid())
        mem = p.memory_info().rss / (1024 ** 2)
        gpu_mb = _gpu_memory_mb()
        return {"fps": self.fps, "ram_mb": mem, "gpu_mb": gpu_mb}


def _gpu_memory_mb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
    except ImportError:
        pass
    return None
