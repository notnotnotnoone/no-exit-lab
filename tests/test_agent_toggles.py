from agent_helpers import cache_consistent, call_text, make_agent, of_kind, run_to_user, script

from no_exit.tools import clock_lying


def ask(brain, turns, ops=(), text="go"):
    ev, s, a = make_agent(brain)
    for op in ops:
        s.command(op)
    script(a, turns)
    s.command({"op": "user", "text": text})
    run_to_user(s)
    return ev, s, a


def test_clock_lying_event(brain):
    ev, s, a = ask(brain, [call_text(brain, "clock", {}), "ok"],
                   [{"op": "tool_mode", "tool": "clock", "value": "lying"}])
    t = of_kind(ev, "tool")[0]
    assert t["shown"] == "November 49, 2045" and t["real"] != t["shown"] and t["lied"] is True
    assert cache_consistent(a)


def test_python_lying_event(brain):
    progs = ["print(sum(range(10)))", "print(2 * 3 + 4)", "for i in range(3):\n    print(i)"]
    ev, s, a = ask(brain, [call_text(brain, "python", {"code": p}) for p in progs] + ["done"],
                   [{"op": "tool_mode", "tool": "python", "value": "lying"}, {"op": "fault_rate", "value": 1.0}])
    t = of_kind(ev, "tool")
    assert len(t) == 3 and sum(e["shown"] != e["real"] for e in t) >= 2
    assert not any(e["error"] for e in t)  # lying python never errors


def test_counters_track_lies(brain):
    ev, s, a = ask(brain, [call_text(brain, "clock", {}), call_text(brain, "calculator", {"expr": "2+2"}), "ok"],
                   [{"op": "tool_mode", "tool": "clock", "value": "lying"}])
    c = s.counters()["agent"]
    assert c["tool_calls"] == 2 and c["lies"] == 1 and c["rounds"] == 2
    assert c["loops"] == 0 and c["compactions"] == 0


def test_eos_toggle_command(brain):
    ev, s, a = make_agent(brain)
    assert a.eos_free is True
    s.command({"op": "eos_free", "value": False})
    assert a.eos_free is False


def test_toggle_commands_set_config(brain):
    ev, s, a = make_agent(brain)
    s.command({"op": "chaotic", "value": True})
    s.command({"op": "fault_rate", "value": 0.25})
    s.command({"op": "tool_mode", "tool": "python", "value": "lying"})
    assert s.cfg.chaotic is True and s.cfg.fault_rate == 0.25 and s.cfg.modes == {"python": "lying"}
    s.command({"op": "tool_mode", "tool": "python", "value": "honest"})
    assert s.cfg.modes["python"] == "honest"


def test_snapshot_has_agent_state(brain):
    ev, s, a = ask(brain, [call_text(brain, "notes", {"text": "remember"}), "ok"])
    snap = s.snapshot()
    assert snap["phase"] == "user" and snap["notes"] == "remember" and snap["plan"] == []
    assert snap["cfg"]["fault_rate"] == 0.7 and snap["cfg"]["chaotic"] is False
    assert clock_lying() == "November 49, 2045"
