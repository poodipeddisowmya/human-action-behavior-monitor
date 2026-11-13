from src.action_recognition.action_classifier import ActionPrediction
from src.temporal.smoother import TemporalSmoother


def test_majority_vote_kills_flicker():
    sm = TemporalSmoother(window=5, min_confidence=0.5)
    seq = ["standing", "sitting", "standing", "standing", "standing"]
    out = None
    for label in seq:
        out = sm.smooth(1, ActionPrediction(label, 0.9))
    assert out is not None
    assert out.label == "standing"


def test_always_returns_a_label_before_window_fills():
    """A tracked person must never be left unlabelled.

    Returning None while the window filled produced an empty "- 0%" box in
    the UI, which reads as a broken detector. The smoother now passes the
    current read through, discounted, until it has enough history to vote.
    """
    sm = TemporalSmoother(window=5, min_confidence=0.5)
    out = sm.smooth(1, ActionPrediction("walking", 0.8))
    assert out is not None
    assert out.label == "walking"
    assert out.confidence < 0.8   # discounted because it is unsmoothed


def test_low_confidence_is_reported_not_dropped():
    """Uncertainty shows up as a low number, never as missing output."""
    sm = TemporalSmoother(window=3, min_confidence=0.6)
    out = sm.smooth(1, ActionPrediction("walking", 0.2))
    assert out is not None
    assert out.label == "walking"
    assert out.confidence <= 0.2


def test_split_vote_lowers_confidence():
    """A bare majority should read as less certain than a unanimous window."""
    sm = TemporalSmoother(window=4, min_confidence=0.0)
    for label in ["standing", "walking", "standing", "walking"]:
        out = sm.smooth(7, ActionPrediction(label, 0.8))
    assert out is not None
    assert out.confidence < 0.8
