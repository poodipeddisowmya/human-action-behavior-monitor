"""Re-uploading the same clip must not spawn foo_1.mp4, foo_2.mp4, ...

This is a regression test for a real mess: the upload route used to pick a
destination by testing whether the *name* was free, so every re-upload of the
same WhatsApp video wrote another byte-identical copy. Seven of them
accumulated in data/samples/ before anyone noticed.
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from config import settings
from src.api.main import app

CLIP = b"\x00\x00\x00\x18ftypmp42" + b"fake-video-payload" * 64


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    (tmp_path / "samples").mkdir()
    with TestClient(app) as c:
        yield c


def _upload(client, name: str, payload: bytes):
    return client.post(
        "/api/sources/upload",
        files={"file": (name, io.BytesIO(payload), "video/mp4")},
        params={"switch": False},
    )


def test_same_content_twice_reuses_one_file(client):
    first = _upload(client, "clip.mp4", CLIP)
    second = _upload(client, "clip.mp4", CLIP)

    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["filename"] == "clip.mp4"
    assert second.json()["filename"] == "clip.mp4", "re-upload created a copy"
    assert first.json()["deduplicated"] is False
    assert second.json()["deduplicated"] is True

    samples = settings.data_dir / "samples"
    assert [f.name for f in samples.iterdir()] == ["clip.mp4"]


def test_different_content_same_name_still_kept_separately(client):
    """A genuinely different video must not be silently dropped or clobbered."""
    _upload(client, "clip.mp4", CLIP)
    resp = _upload(client, "clip.mp4", CLIP + b"different-tail")

    assert resp.json()["filename"] == "clip_1.mp4"
    assert resp.json()["deduplicated"] is False

    samples = settings.data_dir / "samples"
    assert sorted(f.name for f in samples.iterdir()) == ["clip.mp4", "clip_1.mp4"]
    assert (samples / "clip.mp4").read_bytes() == CLIP, "original was overwritten"


def test_no_partial_file_left_behind(client):
    _upload(client, "clip.mp4", CLIP)
    _upload(client, "clip.mp4", CLIP)
    leftovers = list((settings.data_dir / "samples").glob("*.part"))
    assert not leftovers, f"temp files not cleaned up: {leftovers}"
