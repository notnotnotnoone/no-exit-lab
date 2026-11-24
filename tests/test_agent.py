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
