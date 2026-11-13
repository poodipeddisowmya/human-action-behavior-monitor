from datetime import datetime, timedelta

from src.temporal.event_generator import EventGenerator
from src.temporal.smoother import SmoothedPrediction


def test_event_closed_on_label_change():
    captured = []
    gen = EventGenerator(on_event=captured.append)

    t0 = datetime(2026, 1, 1, 10, 0, 0)
    gen.update(1, SmoothedPrediction("standing", 0.9), t0)
    gen.update(1, SmoothedPrediction("standing", 0.9), t0 + timedelta(seconds=1))
    gen.update(1, SmoothedPrediction("walking",  0.9), t0 + timedelta(seconds=2))

    assert len(captured) == 1
    assert captured[0].label == "standing"
    assert captured[0].duration_s == 2.0


def test_close_person_flushes():
    captured = []
    gen = EventGenerator(on_event=captured.append)
    t0 = datetime(2026, 1, 1, 10, 0, 0)
    gen.update(2, SmoothedPrediction("sitting", 0.8), t0)
    gen.close_person(2, t0 + timedelta(seconds=5))
    assert captured[0].label == "sitting"
    assert captured[0].duration_s == 5.0
