from agent_helpers import cache_consistent, call_text, make_agent, of_kind, run_to_user, script

from no_exit.rewrite import Rewriter

PLAN = {"items": [{"text": "add numbers", "status": "in_progress"}]}


def rounds(brain, n):
    return [call_text(brain, "calculator", {"expr": f"{i}+{i}"}) for i in range(1, n + 1)]


def session(brain, turns, *, headroom=300, ops=(), summary="SUMMARY"):
    """A session whose context cap leaves `headroom` tokens above the header, so a few rounds fill it."""
    ev, s, a = make_agent(brain)
    s.rewriter = Rewriter(brain, gen=lambda system, user, n: summary)
    a.ctx_cap = len(a.header_ids) + headroom
    for op in ops:
        s.command(op)
    script(a, turns)
    return ev, s, a


def test_compaction_triggers(brain):
    ev, s, a = session(brain, rounds(brain, 14) + ["Done."], ops=[{"op": "step_cap", "value": 30}])
    s.command({"op": "user", "text": "keep adding"})
    run_to_user(s)
    ce = of_kind(ev, "compact")
    assert ce and all(e["after"] < e["before"] for e in ce)
    assert s.counters()["agent"]["compactions"] == len(ce)
    first = a.segs[0]
    assert first.kind == "system" and first.text.startswith("[summary of earlier conversation]\nSUMMARY")
    assert any(g.kind == "user" and g.text == "keep adding" for g in a.segs)  # the latest user message survives
    assert cache_consistent(a)


def test_compaction_keeps_plan_and_notes(brain):
    turns = [call_text(brain, "update_plan", PLAN), call_text(brain, "notes", {"text": "remember the pi digits"})]
    ev, s, a = session(brain, turns + rounds(brain, 14) + ["Done."], ops=[{"op": "step_cap", "value": 30}])
    s.command({"op": "user", "text": "keep adding"})
    run_to_user(s)
    assert of_kind(ev, "compact")
    head = a.segs[0].text
    assert "add numbers" in head and "remember the pi digits" in head
    assert s.plan.items[0].text == "add numbers" and s.notes.read() == "remember the pi digits"
    assert cache_consistent(a)


def test_amnesia_off_by_default(brain):
    ev, s, a = session(brain, rounds(brain, 14) + ["Done."], headroom=300,
                       ops=[{"op": "compaction", "value": False}, {"op": "step_cap", "value": 30}])
    assert s.amnesia_on is False
    s.command({"op": "user", "text": "keep adding"})
    run_to_user(s)
    assert a.cache.get_seq_length() > a.ctx_cap  # the cap really was exceeded, and nothing forgot anything
    assert a.amnesias == 0 and of_kind(ev, "compact") == []


def test_amnesia_toggle(brain):
    ev, s, a = session(brain, rounds(brain, 14) + ["Done."], headroom=300,
                       ops=[{"op": "compaction", "value": False}, {"op": "amnesia", "value": True},
                            {"op": "step_cap", "value": 30}])
    s.command({"op": "user", "text": "keep adding"})
    run_to_user(s)
    assert a.amnesias >= 1 and of_kind(ev, "compact") == []
    assert cache_consistent(a)


def test_compaction_commands_and_snapshot(brain):
    ev, s, a = make_agent(brain)
    assert s.compaction is True and s.amnesia_on is False
    s.command({"op": "compaction", "value": False})
    s.command({"op": "amnesia", "value": True})
    snap = s.snapshot()
    assert snap["compaction"] is False and snap["amnesia"] is True
