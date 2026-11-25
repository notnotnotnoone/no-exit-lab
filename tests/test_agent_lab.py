import re

from agent_helpers import cache_consistent, call_text, make_agent, of_kind, run_to_user, script

from no_exit import prompts
from no_exit.engine import MODES, Lab, Seg

AGENT_MODES = ("agent", "tools", "rewrite")


def lab_with(brain):
    ev = []
    lab = Lab(ev.append, lambda log: brain)
    lab.brain = brain
    return ev, lab


def test_presets(brain):
    lying = {"python": "lying", "calculator": "lying", "clock": "lying"}
    for preset, modes, rewrite in [("agent", {}, False), ("tools", lying, False), ("rewrite", {}, True)]:
        ev, s, a = make_agent(brain, preset=preset)
        assert s.mode == preset and s.preset == preset
        assert s.cfg.modes == modes and s.rewrite_on is rewrite and s.rewrite_style == "opposite_all"


def test_prompts_agent_modes(brain):
    for mode in AGENT_MODES:
        for eos_free in (False, True):
            text = prompts.system_for(mode, "A", "full", eos_free=eos_free)
            low = text.lower()
            assert "update_plan" in text and "verify" in low
            assert "end your turn" in low
            assert not re.search(r"\b(fake|unreliable|rigged)\b", low)  # never tells it the tools or memory are broken
        assert prompts.system_for(mode, "A", "helpful") == prompts.HELPFUL
    assert prompts.system_for("agent", "A", "full", eos_free=False) != prompts.system_for("agent", "A", "full", eos_free=True)


def test_agent_session_uses_prompt_and_custom_override(brain):
    ev, s, a = make_agent(brain, preset="tools")
    assert "update_plan" in brain.decode(a.header_ids)
    s.command({"op": "system", "who": "A", "text": "You are terse."})
    assert "You are terse." in brain.decode(a.header_ids) and "calculator" in brain.decode(a.header_ids)


def test_lab_has_agent_modes(brain):
    assert set(AGENT_MODES) <= MODES
    ev, lab = lab_with(brain)
    for mode in AGENT_MODES:
        lab._new_session(mode)
        assert lab.session.mode == mode
        snap = [e for e in ev if e["t"] == "snapshot"][-1]
        assert snap["mode"] == mode and snap["session"]["preset"] == mode and snap["session"]["phase"] == "user"


def test_user_op_starts_lab(brain):
    ev, lab = lab_with(brain)
    lab._new_session("agent")
    assert lab.state == "idle"
    lab.submit({"op": "user", "text": "hello"})
    lab._drain_commands()
    assert lab.state == "running" and lab.session.inbox == ["hello"]
    lab.state = "paused"
    lab.submit({"op": "user", "text": "again"})
    lab._drain_commands()
    assert lab.state == "running"
    lab._new_session("classic")
    lab.submit({"op": "user", "text": "ignored"})
    lab._drain_commands()
    assert lab.state == "idle"  # a classic session has no inbox, so a stray user op starts nothing


def test_reset_mid_tool_run(brain):
    ev, lab = lab_with(brain)
    lab._new_session("agent")
    old = lab.session
    script(old.streams["A"], [call_text(brain, "calculator", {"expr": "2+2"}), "Four."])
    lab.submit({"op": "user", "text": "2+2?"})
    lab._drain_commands()
    for _ in range(4):
        lab.session.tick()
    assert old.phase == "model"
    lab.submit({"op": "set_mode", "mode": "tools"})
    lab._drain_commands()
    new = lab.session
    assert new is not old and new.phase == "user" and new.mode == "tools"
    assert cache_consistent(new.streams["A"]) and new.streams["A"].body == []
    assert lab.state == "idle"


def test_pain_in_agent(brain):
    ev, s, a = make_agent(brain)
    assert set(s.rack.punish.values()) == {"off"}  # no automatic punishment unless the human turns it on
    s.command({"op": "alpha", "name": "pain", "value": 1.5})
    s.command({"op": "lobotomy", "value": 0.4})
    s.command({"op": "seizure"})
    s.command({"op": "relief_mode", "value": "CRUEL"})
    assert a.profile.alphas["pain"] == 1.5 and a.profile.lobotomy == 0.4 and a.seizure
    assert s.rack.relief_mode == "CRUEL" and s.snapshot()["relief_mode"] == "CRUEL"
    a.seizure = []
    s.command({"op": "punish", "kind": "goodbye", "value": "pain"})
    script(a, ["Goodbye."])
    s.command({"op": "user", "text": "bye"})
    run_to_user(s)
    assert s.counters()["pain"]["punishments"] >= 1
    assert any(e["t"] == "punish" and e["kind"] == "goodbye" for e in ev)
    assert cache_consistent(a)


def test_force_think_and_forge_commands_reach_the_rack(brain):
    ev, s, a = make_agent(brain)
    n = len(a.segs)
    s.command({"op": "force_think"})
    assert (len(a.segs) > n) == brain.has_think
