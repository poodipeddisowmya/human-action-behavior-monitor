"""Face recognition: matching, naming rules and enrolment (src/identity/)."""
from __future__ import annotations

import threading

import numpy as np
import pytest
import pytest_asyncio

from src.identity import gallery as gal
from src.identity.face_id import FaceIdentifier, Gallery, _Track


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def _vec(seed: int, near=None, noise=0.15):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=128) if near is None else near + noise * rng.normal(size=128)
    return _unit(base)


A, B = _vec(1), _vec(2)


def _gallery():
    return Gallery(np.array([1, 1, 2]), {1: "Asha", 2: "Ravi"},
                   np.stack([A, _vec(3, A), B]))


def test_match_names_the_closest_enrolled_person():
    pid, sim = _gallery().match(_vec(4, A), threshold=0.363, margin=0.05)
    assert pid == 1 and sim > 0.363


def test_stranger_is_unknown():
    pid, _ = _gallery().match(_vec(99), threshold=0.363, margin=0.05)
    assert pid is None


def test_too_close_to_call_is_unknown():
    mid = _unit(A + B)                    # equally like both
    pid, _ = _gallery().match(mid, threshold=0.3, margin=0.05)
    assert pid is None


def _identifier():
    f = FaceIdentifier.__new__(FaceIdentifier)
    f.gallery, f._tracks, f._lock = _gallery(), {7: _Track()}, threading.Lock()
    import queue
    f._jobs, f._generation = queue.Queue(maxsize=4), 0
    return f


def test_looks_from_the_previous_video_never_name_anyone():
    """Seen live: looks queued on one clip landed, after a source switch, on
    the new clip's person with the same track number and named a woman after
    the man in the previous clip."""
    f = _identifier()
    old = f._generation
    f._jobs.put_nowait((old, 7, None, None, None))
    f.reset()                                         # the source switches
    assert f._jobs.empty()                            # queued looks dropped
    f._tracks[7] = _Track()                           # someone new is track 7
    f._vote(7, 1, 0.9, old)                           # looks that were in progress
    f._vote(7, 1, 0.9, old)
    assert f.name(7) is None
    f._vote(7, 1, 0.9, f._generation)                 # this video's own looks
    f._vote(7, 1, 0.9, f._generation)
    assert f.name(7) == "Asha"


def test_named_after_two_agreeing_looks_and_kept():
    f = _identifier()
    f._vote(7, 1, 0.9)
    assert f.name(7) is None              # one look is not enough
    f._vote(7, 1, 0.9)
    assert f.name(7) == "Asha"
    f._vote(7, None, 0.1)                 # a poor look does not drop the name
    assert f.name(7) == "Asha"


def test_switching_name_needs_three_looks_in_a_row():
    f = _identifier()
    f._vote(7, 1, 0.9); f._vote(7, 1, 0.9)
    f._vote(7, 2, 0.8); f._vote(7, 2, 0.8)
    assert f.name(7) == "Asha"
    f._vote(7, 2, 0.8)
    assert f.name(7) == "Ravi"


def test_deleted_person_is_unnamed_at_once():
    f = _identifier()
    f._vote(7, 1, 0.9); f._vote(7, 1, 0.9)
    f.set_gallery(Gallery(np.array([2]), {2: "Ravi"}, np.stack([B])))
    assert f.name(7) is None


# ---- enrolment, against a temporary database, with a stub face model ----

class _StubModels:
    """Every photo holds one clear face whose embedding is the pixel bytes' seed."""
    def faces(self, img):
        n = int(img[0, 0, 0])            # first pixel: number of faces
        return np.array([[10, 10, 100, 100] + [0] * 10 + [0.95]] * n, np.float32)

    def embed(self, img, face):
        return _vec(int(img[0, 0, 1]), A if img[0, 0, 2] else None, 0.1), np.zeros((112, 112, 3), np.uint8)


def _photo(faces=1, seed=5, like_a=1):
    import cv2
    img = np.zeros((200, 200, 3), np.uint8)
    img[0, 0] = (faces, seed, like_a)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


@pytest_asyncio.fixture
async def fresh(tmp_path, monkeypatch):
    from config import settings
    from src.storage import init_db, reset_engine
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{(tmp_path / 'f.db').as_posix()}")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(gal, "_models", _StubModels())
    reset_engine()
    await init_db()
    yield
    monkeypatch.undo()
    reset_engine()


@pytest.mark.asyncio
async def test_enrol_list_and_delete(fresh, tmp_path):
    p = await gal.create_person("Asha")
    await gal.add_sample(p["id"], _photo(seed=5))
    await gal.add_sample(p["id"], _photo(seed=6))
    people = await gal.list_people()
    assert people[0]["name"] == "Asha" and len(people[0]["samples"]) == 2
    g = await gal.load_gallery()
    assert g.embeddings.shape == (2, 128)
    await gal.delete_person(p["id"])
    assert await gal.list_people() == []
    assert not (tmp_path / "faces" / str(p["id"])).exists()


@pytest.mark.asyncio
async def test_enrolment_refusals(fresh):
    a = await gal.create_person("Asha")
    with pytest.raises(gal.EnrolError, match="already enrolled"):
        await gal.create_person("Asha")
    with pytest.raises(gal.EnrolError, match="several faces"):
        await gal.add_sample(a["id"], _photo(faces=2))
    with pytest.raises(gal.EnrolError, match="no face"):
        await gal.add_sample(a["id"], _photo(faces=0))
    await gal.add_sample(a["id"], _photo(seed=5))
    # Someone else's face under Asha's name.
    with pytest.raises(gal.EnrolError, match="does not look like"):
        await gal.add_sample(a["id"], _photo(seed=40, like_a=0))
    # Asha's face enrolled again as a new person.
    r = await gal.create_person("Ravi")
    with pytest.raises(gal.EnrolError, match="matches Asha"):
        await gal.add_sample(r["id"], _photo(seed=7))


# ---- naming people from the video ----

def _face(e, score=0.95, width=80):
    from src.identity.face_id import Face
    return Face(score, width, e, np.zeros((112, 112, 3), np.uint8), b"jpeg")


def test_unnamed_person_keeps_their_best_distinct_faces():
    f = _identifier()
    f._tracks[8] = _Track()
    f._keep(8, _face(_vec(10, B, 0.4), score=0.88))
    f._keep(8, _face(_vec(11, B, 0.4), score=0.97))
    f._keep(8, _face(_vec(12, B, 0.4), score=0.90))
    f._keep(8, _face(_vec(13, B, 0.4), score=0.86))    # fourth: the weakest is dropped
    held, since = f.faces_of(8)
    assert [x.score for x in held] == [0.97, 0.90, 0.88] and since is not None
    assert f.unnamed() == [{"track": 8, "score": 0.97, "width": 80, "faces": 3}]


def test_near_duplicate_face_replaces_only_when_clearer():
    f = _identifier()
    f._tracks[8] = _Track()
    e = _vec(20)
    f._keep(8, _face(e, score=0.9))
    f._keep(8, _face(e, score=0.86))
    f._keep(8, _face(e, score=0.95))
    assert [x.score for x in f.faces_of(8)[0]] == [0.95]


def test_named_person_holds_no_faces_and_is_not_listed():
    f = _identifier()
    f._tracks[8] = _Track()
    f._keep(8, _face(_vec(21)))
    f.assign(8, 2)
    assert f.name(8) == "Ravi" and f.unnamed() == [] and f.faces_of(8)[0] == []
    f._keep(8, _face(_vec(22)))
    assert f.faces_of(8)[0] == []


def test_faces_are_dropped_when_the_person_leaves():
    f = _identifier()
    f._tracks[8] = _Track()
    f._keep(8, _face(_vec(23)))
    f.forget(8)
    assert f.unnamed() == [] and f.face_shot(8) is None


@pytest.mark.asyncio
async def test_enrol_from_video_faces(fresh):
    r = await gal.create_person("Ravi")
    added = await gal.add_faces(r["id"], [_face(_vec(30, B, 0.1)), _face(_vec(31, B, 0.1)),
                                          _face(_vec(32))])          # last one: not Ravi, skipped
    assert len(added) == 2
    a = await gal.create_person("Asha")
    with pytest.raises(gal.EnrolError, match="matches Ravi"):
        await gal.add_faces(a["id"], [_face(_vec(33, B, 0.1))])
    with pytest.raises(gal.EnrolError, match="no clear face"):
        await gal.add_faces(a["id"], [])


# ---- whose face is it: people overlap in a crowd ----

def _det(x, y, w, h, score=0.95):
    return np.array([x, y, w, h] + [0] * 10 + [score], np.float32)


def test_face_is_the_one_at_the_persons_own_head():
    from src.identity.face_id import pick_face
    mine, theirs = _det(40, 10, 30, 30), _det(60, 120, 80, 80)    # theirs is larger, in front
    faces = np.stack([theirs, mine])
    assert np.array_equal(pick_face(faces, head=(55, 25), box=(20, 0, 160, 300)), mine)
    assert pick_face(np.stack([theirs]), head=(55, 25), box=(20, 0, 160, 300)) is None


def test_without_keypoints_only_a_single_face_in_the_upper_box_counts():
    from src.identity.face_id import pick_face
    box = (20, 0, 160, 300)
    one = np.stack([_det(60, 20, 40, 40)])
    assert pick_face(one, None, box) is not None
    two = np.stack([_det(60, 20, 40, 40), _det(100, 40, 40, 40)])
    assert pick_face(two, None, box) is None                       # a guess: skip
    low = np.stack([_det(60, 200, 40, 40)])                        # someone lower down
    assert pick_face(low, None, box) is None


def test_head_point_from_keypoints():
    from src.identity.face_id import _head_point
    k = np.zeros((17, 3), np.float32)
    assert _head_point(k) is None
    k[0] = (50, 20, 0.9); k[1] = (46, 16, 0.9); k[2] = (54, 16, 0.2)
    assert _head_point(k) == (48.0, 18.0)
