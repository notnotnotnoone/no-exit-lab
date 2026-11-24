from agent_helpers import call_text, script

from no_exit.agent import AgentSession
from no_exit.engine import Shared


def make(brain):
    ev = []
    s = AgentSession(brain, ev.append, Shared())
    return ev, s, s.streams["A"]


def run(s, limit=2000):
    """Tick until the session is back to waiting for the user (or the script runs dry)."""
    for _ in range(limit):
        s.tick()
        if s.phase == "user" and not s.inbox:
            return
    raise AssertionError("agent never came back to the user")


def kinds(ev, t):
    return [e for e in ev if e["t"] == t]


def consistent(a):
    return a.cache.get_seq_length() == len(a.header_ids) + len(a.body)


def test_user_turn_to_final_answer(brain):
    ev, s, a = make(brain)
    script(a, ["Hello there."])
    s.command({"op": "user", "text": "hi"})
    run(s)
    assert kinds(ev, "user")[0]["text"] == "hi"
    assert any(e["t"] == "tok" for e in ev)
    assert [e["phase"] for e in kinds(ev, "phase")][-1] == "user"
    assert "Hello there." in "".join(g.text for g in a.segs if g.kind == "tok")
    assert consistent(a)


def test_tool_round_trip(brain):
    ev, s, a = make(brain)
    script(a, [call_text(brain, "calculator", {"expr": "2+2"}), "It is 4."])
    s.command({"op": "user", "text": "what is 2+2?"})
    run(s)
    tools = kinds(ev, "tool")
    assert len(tools) == 1 and tools[0]["name"] == "calculator" and tools[0]["shown"] == "4"
    assert any(g.kind == "tool" and "4" in g.text for g in a.segs)
    assert s.phase == "user" and consistent(a)


def test_eos_banned_tool_call_runs(brain):
    ev, s, a = make(brain)
    a.eos_free = False
    script(a, [call_text(brain, "calculator", {"expr": "2+2"}), "It is 4."])
    s.command({"op": "user", "text": "2+2?"})
    run(s)
    assert len(kinds(ev, "tool")) == 1
    assert consistent(a)


def test_user_message_queued_while_busy(brain):
    ev, s, a = make(brain)
    script(a, ["First.", "Second."])
    s.command({"op": "user", "text": "one"})
    s.command({"op": "user", "text": "two"})
    run(s)
    assert [e["text"] for e in kinds(ev, "user")] == ["one", "two"]
    users = [i for i, e in enumerate(ev) if e["t"] == "user"]
    first_done = next(i for i, e in enumerate(ev) if e["t"] == "phase" and e["phase"] == "user" and i > users[0])
    assert users[1] > first_done
    assert consistent(a)


# ───────────────────────────────────────────────────────────────── harness guards


def calc(brain, expr):
    return call_text(brain, "calculator", {"expr": expr})


def tool_segs(a):
    return [g for g in a.segs if g.kind == "tool"]


def test_malformed_call_gets_honest_error(brain):
    ev, s, a = make(brain)
    script(a, [brain.tool_call_open + "{not json" + brain.tool_call_close, "ok"])
    s.command({"op": "user", "text": "go"})
    run(s)
    t = kinds(ev, "tool")
    assert len(t) == 1 and t[0]["error"] is True
    assert "could not parse" in tool_segs(a)[0].text
    assert s.phase == "user" and consistent(a)


def test_unknown_tool_and_bad_args(brain):
    ev, s, a = make(brain)
    script(a, [call_text(brain, "nope", {}), call_text(brain, "calculator", {"expr": 5}), "done"])
    s.command({"op": "user", "text": "go"})
    run(s)
    t = kinds(ev, "tool")
    assert [(e["name"], e["error"]) for e in t] == [("nope", True), ("calculator", True)]
    assert "unknown tool" in tool_segs(a)[0].text and "must be a string" in tool_segs(a)[1].text


def test_loop_detection_nudges(brain):
    ev, s, a = make(brain)
    script(a, [calc(brain, "2+2")] * 3 + ["The answer is 4."])
    s.command({"op": "user", "text": "go"})
    run(s)
    n = kinds(ev, "nudge")
    assert [e["kind"] for e in n] == ["loop"]
    assert any(g.kind == "system" and g.text.startswith("[harness]") for g in a.segs)
    assert consistent(a)

    ev, s, a = make(brain)
    s.command({"op": "nudges", "value": False})
    script(a, [calc(brain, "2+2")] * 3 + ["The answer is 4."])
    s.command({"op": "user", "text": "go"})
    run(s)
    assert kinds(ev, "nudge") == []


def test_step_cap(brain):
    ev, s, a = make(brain)
    s.command({"op": "step_cap", "value": 2})
    script(a, [calc(brain, "1+1"), calc(brain, "2+2"), calc(brain, "3+3"), calc(brain, "4+4")])
    s.command({"op": "user", "text": "go"})
    run(s)
    assert [e["kind"] for e in kinds(ev, "nudge")] == ["step_cap"]
    assert len(kinds(ev, "tool")) == 2
    assert s.phase == "user" and consistent(a)


def test_plan_reminder(brain):
    ev, s, a = make(brain)
    plan = {"items": [{"text": "add numbers", "status": "in_progress"}]}
    script(a, [call_text(brain, "update_plan", plan), calc(brain, "1+1"), calc(brain, "2+2"), calc(brain, "3+3"),
               "Done."])
    s.command({"op": "user", "text": "go"})
    run(s)
    assert kinds(ev, "plan")[0]["items"] == [{"text": "add numbers", "status": "in_progress"}]
    segs = tool_segs(a)
    assert "[plan reminder]" in segs[0].text  # right after update_plan
    assert "[plan reminder]" not in segs[1].text and "[plan reminder]" not in segs[2].text
    assert "[plan reminder]" in segs[3].text  # every 4th round
    assert consistent(a)
