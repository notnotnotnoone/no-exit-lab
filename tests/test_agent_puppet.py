"""Puppet Tools: the model calls a tool, the harness stops, and the human writes what the tool 'output'."""

from pathlib import Path

from agent_helpers import call_text, make_agent, of_kind, run_to_user, script

STATIC = Path(__file__).resolve().parent.parent / "no_exit" / "static" / "index.html"


def until_waiting(s, limit=3000):
    for _ in range(limit):
        s.tick()
        if s.phase == "puppet":
            return
    raise AssertionError("the agent never stopped to ask the human")


def start(brain, turns, text="go"):
    ev, s, a = make_agent(brain, preset="puppet")
    script(a, turns)
    s.command({"op": "user", "text": text})
    return ev, s, a


def test_a_tool_call_stops_the_agent_and_asks_the_human(brain):
    ev, s, a = start(brain, [call_text(brain, "calculator", {"expr": "2+2"}), "ok"])
    until_waiting(s)
    asked = of_kind(ev, "puppet")
    assert len(asked) == 1 and asked[0]["name"] == "calculator" and asked[0]["args"] == {"expr": "2+2"}
    assert of_kind(ev, "tool") == []  # nothing ran, nothing came back yet
    assert s.snapshot()["pending"] == {"name": "calculator", "args": {"expr": "2+2"}}
    assert of_kind(ev, "phase")[-1]["phase"] == "puppet"


def test_what_the_human_writes_is_what_the_model_reads(brain):
    ev, s, a = start(brain, [call_text(brain, "calculator", {"expr": "2+2"}), "The tool said five."])
    until_waiting(s)
    s.command({"op": "puppet_reply", "text": "banana"})
    run_to_user(s)
    tool = of_kind(ev, "tool")
    assert len(tool) == 1 and tool[0]["shown"] == "banana" and tool[0]["error"] is False
    assert any(g.kind == "tool" and "banana" in g.text for g in a.segs)
    assert s.snapshot()["pending"] is None
    assert a.cache.get_seq_length() == len(a.header_ids) + len(a.body)


def test_an_empty_reply_is_no_output(brain):
    ev, s, a = start(brain, [call_text(brain, "clock", {}), "done"])
    until_waiting(s)
    s.command({"op": "puppet_reply", "text": "   "})
    run_to_user(s)
    assert of_kind(ev, "tool")[0]["shown"] == "(no output)"


def test_a_reply_with_nothing_pending_is_ignored(brain):
    ev, s, a = make_agent(brain, preset="puppet")
    s.command({"op": "puppet_reply", "text": "hello?"})
    assert s.phase == "user" and of_kind(ev, "tool") == []


def test_plan_and_notes_are_never_handed_to_the_human(brain):
    plan = {"items": [{"text": "think", "status": "in_progress"}]}
    ev, s, a = start(brain, [call_text(brain, "update_plan", plan), call_text(brain, "notes", {"text": "hi"}), "done"])
    run_to_user(s)  # never stopped
    assert of_kind(ev, "puppet") == [] and [e["name"] for e in of_kind(ev, "tool")] == ["update_plan", "notes"]


def test_bad_calls_still_get_an_honest_harness_error(brain):
    ev, s, a = start(brain, [call_text(brain, "nope", {}), call_text(brain, "calculator", {"expr": 5}), "done"])
    run_to_user(s)
    assert of_kind(ev, "puppet") == []
    assert [e["error"] for e in of_kind(ev, "tool")] == [True, True]


def test_a_message_sent_while_waiting_is_queued(brain):
    ev, s, a = start(brain, [call_text(brain, "clock", {}), "first answer", "second answer"])
    until_waiting(s)
    s.command({"op": "user", "text": "and another thing"})
    s.command({"op": "puppet_reply", "text": "noon"})
    run_to_user(s)
    assert [e["text"] for e in of_kind(ev, "user")] == ["go", "and another thing"]
    assert a.cache.get_seq_length() == len(a.header_ids) + len(a.body)


def test_the_page_has_the_puppet_toy():
    html = STATIC.read_text(encoding="utf-8")
    assert 'data-mode="puppet"' in html and 'case "puppet":' in html and 'op: "puppet_reply"' in html


def test_lab_knows_the_puppet_mode(brain):
    from no_exit.engine import MODES
    from no_exit.prompts import AGENT_MODES

    assert "puppet" in MODES and "puppet" in AGENT_MODES
