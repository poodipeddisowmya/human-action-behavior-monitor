"""When the detector uses its larger HD input size (src/tracking/person_tracker.py).

Measured: 960 px finds distant people (street scene 1.33 -> 4.0 people per
frame) but slices a close-up into phantom people (one-person kitchen clip
1.0 -> 2.42, captioned "three people"). So HD is used only while the people
in view are small.
"""
from __future__ import annotations

import numpy as np
import pytest

from config import settings
from src.tracking.person_tracker import PersonTracker


@pytest.fixture
def tracker(monkeypatch):
    monkeypatch.setattr(settings.model, "detector_imgsz", 640)
    monkeypatch.setattr(settings.model, "detector_imgsz_hd", 960)
    monkeypatch.setattr(settings.model, "detector_hd_min_width", 720)
    t = PersonTracker.__new__(PersonTracker)     # no model needed for the rule
    t._use_hd, t._tallest = False, 1.0
    return t


HD = np.zeros((720, 1280, 3), np.uint8)
SMALL = np.zeros((240, 320, 3), np.uint8)


def test_first_frame_uses_base_size(tracker):
    assert tracker._imgsz(HD) == 640


def test_small_people_switch_to_hd(tracker):
    tracker._tallest = 0.15                      # distant pedestrians
    assert tracker._imgsz(HD) == 960


def test_close_up_stays_at_base_size(tracker):
    for _ in range(10):
        tracker._tallest = 0.95                  # one person filling the frame
        assert tracker._imgsz(HD) == 640


def test_phantom_large_box_switches_back(tracker):
    tracker._tallest = 0.1
    assert tracker._imgsz(HD) == 960
    tracker._tallest = 0.8                       # a large box appeared at HD
    assert tracker._imgsz(HD) == 640


def test_hysteresis_holds_between_thresholds(tracker):
    tracker._tallest = 0.1
    tracker._imgsz(HD)
    tracker._tallest = 0.40                      # between 0.35 and 0.45
    assert tracker._imgsz(HD) == 960
    tracker._tallest = 0.6
    tracker._imgsz(HD)
    tracker._tallest = 0.40
    assert tracker._imgsz(HD) == 640


def test_low_resolution_video_never_uses_hd(tracker):
    tracker._tallest = 0.05
    assert tracker._imgsz(SMALL) == 640


def test_disabled_without_hd_size(tracker, monkeypatch):
    monkeypatch.setattr(settings.model, "detector_imgsz_hd", 0)
    tracker._tallest = 0.05
    assert tracker._imgsz(HD) == 640
