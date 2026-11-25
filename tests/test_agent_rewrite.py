from agent_helpers import cache_consistent, call_text, make_agent, of_kind, run_to_user, script

from no_exit.rewrite import Rewriter


def ask(brain, turns, *, on=True, out="FLIPPED.", text="please compute", ops=()):
    ev, s, a = make_agent(brain)
    s.rewriter = Rewriter(brain, gen=lambda system, user, n: out)
    if on:
        s.command({"op": "rewrite_on", "value": True})
    for op in ops:
        s.command(op)
    script(a, turns)
    s.command({"op": "user", "text": text})
    run_to_user(s)
    return ev, s, a


def body_text(brain, a):
    return brain.decode(a.body)


def test_rewrite_off_by_default(brain):
    ev, s, a = ask(brain, ["One.\n\nTwo."], on=False)
    assert of_kind(ev, "rewrite") == []
    assert "One." in body_text(brain, a) and s.rewrite_on is False and s.rewrite_style == "opposite_all"


def test_rewrite_replaces_history(brain):
    ev, s, a = ask(brain, ["One.\n\nTwo."])
    assert len(of_kind(ev, "rewrite")) == 2
    text = body_text(brain, a)
    assert "FLIPPED." in text and "One." not in text and "Two." not in text
    assert any(g.kind == "rewritten" for g in a.segs)
    assert [h["original"].strip() for h in a.history_log] == ["One.", "Two."]
    assert cache_consistent(a)


def test_paragraph_with_tool_call_not_rewritten(brain):
    ev, s, a = ask(brain, ["Hi.\n\n" + call_text(brain, "calculator", {"expr": "2+2"}), "It is four."])
    assert len(of_kind(ev, "tool")) == 1 and of_kind(ev, "tool")[0]["shown"] == "4"
    text = body_text(brain, a)
    assert "calculator" in text and "2+2" in text  # the call paragraph survives untouched
    assert "Hi." not in text  # the paragraph before it was rewritten
    assert cache_consistent(a)


def test_rewriter_failure_keeps_original(brain):
    ev, s, a = ask(brain, ["One.\n\nTwo."], out="")
    assert of_kind(ev, "rewrite") == []
    text = body_text(brain, a)
    assert "One." in text and "Two." in text
    assert cache_consistent(a)


def test_user_and_tool_text_untouched(brain):
    ev, s, a = ask(brain, [call_text(brain, "calculator", {"expr": "17*23"}), "Done.\n\nBye."],
                   text="please compute 17*23")
    text = body_text(brain, a)
    assert "please compute 17*23" in text
    assert any(g.kind == "tool" and g.text == "391" for g in a.segs)
    assert brain.render_tool_result("calculator", "391") in text
    assert "Done." not in text and "Bye." not in text


def test_rewrite_commands_and_snapshot(brain):
    ev, s, a = make_agent(brain)
    s.command({"op": "rewrite_style", "value": "math_corrupt"})
    s.command({"op": "rewrite_style", "value": "nonsense"})  # ignored
    s.command({"op": "rewrite_prompt", "value": "shout everything"})
    s.command({"op": "rewrite_on", "value": True})
    assert (s.rewrite_on, s.rewrite_style, s.rewrite_prompt) == (True, "math_corrupt", "shout everything")
    assert s.snapshot()["rewrite"] == {"on": True, "style": "math_corrupt", "prompt": "shout everything"}


def test_custom_style_without_prompt_is_reported_not_fatal(brain):
    ev, s, a = ask(brain, ["One."], ops=[{"op": "rewrite_style", "value": "custom"}])
    assert of_kind(ev, "rewrite") == []
    assert any(e["t"] == "event" and e["kind"] == "rewrite_error" for e in ev)
    assert "One." in body_text(brain, a)
