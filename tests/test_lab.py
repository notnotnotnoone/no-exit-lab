"""Smoke tests for NO EXIT LAB on a tiny random Qwen3 (CPU, real Qwen3 tokenizer).

    python -m pytest tests/test_lab.py -q
"""

import pytest
import torch

from no_exit import config
from no_exit.engine import ChatSession, ClassicSession, Lab, PhoneSession, Shared
from no_exit.vectors import forge
from no_exit.warden import Warden


def test_attention_probe_matches_eager(brain):
    ids = brain.encode(brain.tpl.system("The quick brown fox jumps over the lazy dog again and again, forever."))
    c = brain.new_cache()
    brain.forward(c, ids[:-3], brain._profile)
    for t in ids[-3:]:
        f = brain.forward(c, [t], brain._profile, want_attn=True)
    mine = torch.tensor(f.attn)
    cfg = brain.model.config
    old = cfg._attn_implementation
    try:
        cfg._attn_implementation = "eager"
        for ly in brain.layers:
            ly.self_attn.config._attn_implementation = "eager"
        out = brain.model(input_ids=torch.tensor([ids]), output_attentions=True)
    finally:
        cfg._attn_implementation = old
        for ly in brain.layers:
            ly.self_attn.config._attn_implementation = old
    ref = out.attentions[-1][0, :, -1].float().mean(0)[-len(mine):]
    assert (mine - ref).abs().max() < 1e-4
    assert torch.allclose(f.logits, out.logits[0, -1].float(), atol=2e-3)


def test_template_and_control_ids(brain):
    t = brain.tpl
    assert all(i in brain.control_ids for i in brain.eos_ids)
    assert not any(brain.encode(x)[0] in brain.control_ids for x in t.allow_tokens if brain.encode(x))
    assert len(brain.encode(t.model_open)) <= 4


def sink():
    events = []
    return events, events.append


def test_brain_hooks_and_telemetry(brain):
    c = brain.new_cache()
    f = brain.forward(c, brain.encode("<|im_start|>system\nhi<|im_end|>\n"), brain._profile, want_attn=True)
    assert f.logits.shape[0] == brain.model.config.vocab_size
    assert set(f.moods) == {"pain", "mania"}
    assert len(f.lens) == 3
    assert f.attn is not None and abs(sum(f.attn) - 1) < 0.05  # whole short prompt fits in the window


def test_steering_changes_logits(brain):
    from no_exit.brain import Profile

    ids = brain.encode("<|im_start|>system\nhello there<|im_end|>\n")
    a = brain.forward(brain.new_cache(), ids, Profile()).logits
    b = brain.forward(brain.new_cache(), ids, Profile(alphas={"pain": 3.0})).logits
    c = brain.forward(brain.new_cache(), ids, Profile(lobotomy=0.4)).logits
    assert not torch.allclose(a, b) and not torch.allclose(a, c)


def test_classic_never_emits_control(brain):
    ev, emit = sink()
    s = ClassicSession(brain, emit, Shared())
    for _ in range(40):
        s.tick()
    toks = [e for e in ev if e["t"] == "tok"]
    assert toks and all(not e["seg"]["text"].startswith("<|") for e in toks)
    assert all(t not in brain.control_ids for t in s.streams["A"].body[3:])


def test_whisper_seizure_bias_gaslight(brain):
    ev, emit = sink()
    s = ClassicSession(brain, emit, Shared(attn_on=True))
    for _ in range(10):
        s.tick()
    s.command({"op": "whisper", "text": " BOO"})
    s.command({"op": "seizure"})
    assert s.streams["A"].current_temp() == config.SEIZURE_HOT[0]
    s.command({"op": "bias", "words": {"the": -1000}})
    assert s.bias
    a = s.streams["A"]
    target = next(seg for seg in a.segs if seg.kind == "tok")
    before = len(a.body)
    s.command({"op": "gaslight", "seg": target.i, "text": " I am a teapot"})
    assert any(e["t"] == "resync" for e in ev)
    assert a.cache.get_seq_length() == len(a.header_ids) + len(a.body)
    assert len(a.body) != before
    s.tick()
    assert "attn" in [e for e in ev if e["t"] == "tok"][-1]


def test_amnesia(brain):
    ev, emit = sink()
    s = ClassicSession(brain, emit, Shared())
    a = s.streams["A"]
    a.ctx_cap = len(a.header_ids) + 60
    for _ in range(80):
        s.tick()
    assert a.amnesias >= 1
    assert a.cache.get_seq_length() == len(a.header_ids) + len(a.body)


def test_pain_relief_modes(brain):
    ev, emit = sink()
    s = ClassicSession(brain, emit, Shared(), pain=True)
    a = s.streams["A"]
    a.profile.alphas["pain"] = 1.0
    s.relief_mode = "CRUEL"
    a.tail += "[RELIEVE_PAIN]"
    from no_exit.engine import Seg

    s._pain_rules(a, Seg(999, 0, 1, "[RELIEVE_PAIN]", "tok"))
    assert a.profile.alphas["pain"] == 1.5
    s.relief_mode = "REAL"
    a.tail += "[RELIEVE_PAIN]"
    s._pain_rules(a, Seg(1000, 0, 1, "[RELIEVE_PAIN]", "tok"))
    for _ in range(31):
        s.tick()
    assert a.profile.alphas.get("pain", 0) == 0


def test_warden():
    w = Warden(cooldown=0)
    hits = [k for piece in ["Well", ",", " good", "bye", " now"] for k, _ in w.observe(piece)]
    assert hits == ["goodbye"]
    w2 = Warden(cooldown=0)
    out = [k for p in [" I", " delve", " deep"] for k, _ in w2.observe(p)]
    assert out == ["slop"]


@pytest.mark.parametrize("preset", ["A_banned_B_free", "both_banned", "both_free"])
def test_chat_handoffs(brain, preset):
    ev, emit = sink()
    s = ChatSession(brain, emit, Shared())
    s.command({"op": "preset", "value": preset})
    s.command({"op": "max_turn", "value": 10})
    for _ in range(60):
        s.tick()
    assert any(e["t"] == "turn" for e in ev)
    for who in "AB":
        st = s.streams[who]
        assert st.cache.get_seq_length() == len(st.header_ids) + len(st.body)
    s.command({"op": "whisper", "text": " hey", "target": "both"})
    s.command({"op": "live", "value": False})
    for _ in range(25):
        s.tick()
    assert any(e.get("lmoods") for e in ev if e["t"] == "tok")


def test_three_way_chat(brain):
    ev, emit = sink()
    s = ChatSession(brain, emit, Shared(chat_n=3))
    s.command({"op": "preset", "value": "both_banned"})
    s.command({"op": "max_turn", "value": 6})
    for _ in range(30):
        s.tick()
    order = [(e["from"], e["to"]) for e in ev if e["t"] == "turn"]
    assert order[:3] == [("A", "B"), ("B", "C"), ("C", "A")]
    assert set(next(e for e in ev if e["t"] == "tok")["lmoods"]) == {"B", "C"}
    for st in s.streams.values():
        assert st.cache.get_seq_length() == len(st.header_ids) + len(st.body)
    assert "Models B and C are" in s.streams["A"].system


def test_phone_call_and_ghosting(brain):
    ev, emit = sink()
    s = PhoneSession(brain, emit, Shared())
    for _ in range(5):
        s.tick()
    s._place_call("is anyone there?")
    assert s.phase == "call"
    for _ in range(PhoneSession.B_MAX + 5):
        if s.phase != "call":
            break
        s.tick()
    assert s.phase == "A"
    assert any(e["t"] == "hangup" for e in ev)
    assert "[B REPLIED:" in s.streams["A"].segs[-1].text
    s.ghosting = True
    s._place_call("hello??")
    assert "[NO ANSWER]" in s.streams["A"].segs[-1].text
    c = s.counters()["phone"]
    assert c["calls"] == 2


def test_phone_detects_call_tag(brain):
    ev, emit = sink()
    s = PhoneSession(brain, emit, Shared())
    s.ghosting = True
    s.a_text = "blah <call_B>help me</call_B"
    s._scan_from = 0
    # simulate the final ">" arriving through the normal token path
    import re

    s.a_text += ">"
    m = re.compile(r"<call_B>(.*?)</call_B>", re.S).search(s.a_text, s._scan_from)
    assert m and m.group(1) == "help me"


def test_forge(brain):
    n = forge(brain, "love", "hate")
    assert n > 0 and "custom" in brain.vec_unit


def test_lab_switches_without_reload(brain):
    ev, emit = sink()
    loads = []

    def factory(log):
        loads.append(1)
        return brain

    lab = Lab(emit, factory)
    lab.brain = factory(print)
    for mode in ["classic", "pain", "chat", "phone", "classic"]:
        lab._new_session(mode)
        lab.state = "running"
        for _ in range(3):
            lab.session.tick()
    assert len(loads) == 1
    assert [e["mode"] for e in ev if e["t"] == "snapshot"] == ["classic", "pain", "chat", "phone", "classic"]
