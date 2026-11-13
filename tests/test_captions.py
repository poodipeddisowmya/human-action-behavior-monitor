from src.description.captions import CaptionComposer, compose_sentence


def _ann(pid, label, conf=0.8):
    return {"person_id": pid, "label": label, "confidence": conf}


def test_single_person_with_scene():
    assert compose_sentence(["cooking"], "in a kitchen") == "A person is cooking in a kitchen."


def test_single_person_unknown_activity():
    assert compose_sentence([None], None) == "One person; identifying the activity…"


def test_nobody():
    assert compose_sentence([], None) == ""


def test_group_puts_activities_before_posture():
    s = compose_sentence(["standing", "smoking", "smoking"], "on a street")
    assert s == "Three people on a street: two are smoking, one is standing."


def test_unknown_people_are_counted_not_narrated():
    assert compose_sentence(["eating", None], None) == "Two people: one is eating."
    assert compose_sentence([None, None, None], "in a classroom") == "Three people in a classroom."


def test_head_count_wobble_does_not_reach_the_caption():
    c = CaptionComposer()
    t = 0.0
    group = [_ann(i, "sitting") for i in range(4)]
    for _ in range(50):
        c.update(group, "in a classroom", now=t)
        t += 0.1
    seen = {c.text}
    for k in range(120):                      # a fifth person flickers in and out
        extra = [_ann(9, "detecting…", 0.0)] if (k // 15) % 2 == 0 else []
        c.update(group + extra, "in a classroom", now=t)
        t += 0.1
        seen.add(c.text)
    # At most one change (four -> five, the recent high), never back and forth.
    assert len(seen) <= 2
    assert c.text.startswith("Five people")


def test_low_confidence_label_is_not_narrated():
    c = CaptionComposer()
    for t in range(0, 40):
        c.update([_ann(1, "standing", 0.40)], None, now=t * 0.1)
    assert c.text == "One person; identifying the activity…"


def test_one_frame_flicker_never_reaches_the_caption():
    c = CaptionComposer()
    t = 0.0
    for _ in range(40):
        c.update([_ann(1, "cooking")], "in a kitchen", now=t)
        t += 0.1
    assert c.text == "A person is cooking in a kitchen."
    c.update([_ann(1, "walking")], "in a kitchen", now=t)   # a single stray frame
    t += 0.1
    for _ in range(30):
        c.update([_ann(1, "cooking")], "in a kitchen", now=t)
        t += 0.1
        assert c.text == "A person is cooking in a kitchen."


def test_a_real_change_comes_through_after_the_hold():
    c = CaptionComposer()
    t = 0.0
    for _ in range(40):
        c.update([_ann(1, "sitting")], None, now=t)
        t += 0.1
    for _ in range(40):
        c.update([_ann(1, "smoking")], None, now=t)
        t += 0.1
    assert c.text == "A person is smoking."


def test_everyone_doing_the_same_thing_reads_naturally():
    assert compose_sentence(["using computer", "using computer"], "in an office") == \
        "Two people are working on a computer in an office."


def test_reset_returns_to_starting_text():
    c = CaptionComposer()
    for t in range(40):
        c.update([_ann(1, "eating")], None, now=t * 0.1)
    c.reset()
    assert c.text == "Analysing the scene…"


def test_tracker_renumbering_does_not_stall_the_caption():
    # The tracker hands the same runner a new ID every few frames.
    c = CaptionComposer()
    t = 0.0
    for i in range(40):
        c.update([_ann(100 + i // 3, "running")], None, now=t)
        t += 0.1
    assert c.text == "A person is running."


def test_confirmed_activity_appears_quickly():
    c = CaptionComposer()
    t = 0.0
    for _ in range(30):                       # person present, nothing known yet
        c.update([_ann(1, "detecting…", 0.0)], None, now=t)
        t += 0.1
    assert "identifying" in c.text
    start = t
    while c.text != "A person is cooking." and t - start < 5:
        c.update([_ann(1, "cooking")], None, now=t)
        t += 0.1
    assert c.text == "A person is cooking."
    assert t - start <= 1.2                   # was ~2.5 s with one slow hold


def test_replacing_a_confirmed_activity_still_waits():
    c = CaptionComposer()
    t = 0.0
    for _ in range(40):
        c.update([_ann(1, "cooking")], None, now=t)
        t += 0.1
    for _ in range(12):                       # 1.2 s of something else
        c.update([_ann(1, "eating")], None, now=t)
        t += 0.1
    assert c.text == "A person is cooking."


def test_group_posture_flicker_is_not_narrated():
    c = CaptionComposer()
    t = 0.0
    for _ in range(60):
        anns = [_ann(i, "sitting") for i in range(4)] + [_ann(9, "running", 0.7)]
        c.update(anns, "in a classroom", now=t)
        t += 0.1
    assert c.text == "Five people in a classroom: four are sitting."



def test_object_and_posture_enrich_a_confirmed_activity():
    s = compose_sentence(["drinking"], "in an office", details={"drinking": "cup"},
                         posture="sitting")
    assert s == "A seated person is drinking from a cup in an office."


def test_object_of_another_activity_is_ignored():
    s = compose_sentence(["drinking"], None, details={"eating": "fork"})
    assert s == "A person is drinking."


def test_long_activity_reports_its_duration():
    s = compose_sentence(["using computer"], "in an office",
                         details={"using computer": "laptop"}, duration_s=185)
    assert s == "A person is working on a laptop in an office for about 3 minutes."


def test_group_shared_posture_goes_in_the_head():
    s = compose_sentence(["sitting", "sitting", "using computer", None], "in an office",
                         group_posture="sitting")
    assert s == "Four people seated in an office: one is working on a computer."


def test_glimpsed_object_is_not_narrated():
    c = CaptionComposer()
    t = 0.0
    for k in range(60):
        a = _ann(1, "drinking")
        a["detail"] = "cup" if k == 30 else None      # seen in one frame only
        c.update([a], None, now=t)
        t += 0.1
    assert c.text == "A person is drinking."


def test_steady_object_and_posture_are_narrated():
    c = CaptionComposer()
    t = 0.0
    for _ in range(60):
        a = _ann(1, "drinking")
        a["detail"], a["posture"] = "bottle", "sitting"
        c.update([a], "in an office", now=t)
        t += 0.1
    assert c.text == "A seated person is drinking from a bottle in an office."
