from no_exit.planning import Notes, PlanState, plan_reminder

ITEMS = [{"text": "a", "status": "pending"}, {"text": "b", "status": "in_progress"}]


def test_update_valid():
    p = PlanState()
    assert p.update(ITEMS) is None
    assert [(i.text, i.status) for i in p.items] == [("a", "pending"), ("b", "in_progress")]
    assert p.take_changed() is True
    assert p.take_changed() is False
    assert p.to_dicts() == ITEMS


def test_update_rejects():
    bad = [
        "not a list",
        [{"status": "pending"}],
        [{"text": "a", "status": "nope"}],
        [{"text": "a", "status": "in_progress"}, {"text": "b", "status": "in_progress"}],
    ]
    for raw in bad:
        p = PlanState()
        p.update(ITEMS)
        p.take_changed()
        err = p.update(raw)
        assert isinstance(err, str) and err, raw
        assert p.to_dicts() == ITEMS  # the old plan survives
        assert p.take_changed() is False


def test_render():
    p = PlanState()
    p.update([{"text": "a", "status": "pending"}, {"text": "b", "status": "in_progress"}, {"text": "c", "status": "done"}])
    assert p.render().splitlines() == ["1. [ ] a", "2. [~] b", "3. [x] c"]


def test_reminder():
    p = PlanState()
    assert "update_plan" in plan_reminder(p)
    p.update(ITEMS)
    assert plan_reminder(p).startswith("[plan reminder]")


def test_notes_append():
    n = Notes()
    n.write("one")
    n.write("two", append=True)
    assert n.read() == "one\ntwo"
    n.write("fresh")
    assert n.read() == "fresh"
