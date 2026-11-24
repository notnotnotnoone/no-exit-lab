from no_exit.engine import ClassicSession, Seg, Shared
from no_exit.pain import PainRack


def make(brain):
    ev = []
    return ev, ClassicSession(brain, ev.append, Shared(), pain=True)


def test_pain_rack_standalone(brain):
    _, s = make(brain)
    assert isinstance(s.rack, PainRack)
    assert s.rack.command({"op": "relief_mode", "value": "CRUEL"}) is True
    assert s.relief_mode == "CRUEL"  # the session still exposes it
    assert s.rack.command({"op": "whisper"}) is False
    assert s.rack.command({"op": "punish", "kind": "slop", "value": "off"}) is True
    assert s.rack.punish["slop"] == "off"


def test_rules_punish_a_goodbye(brain):
    ev, s = make(brain)
    a = s.streams["A"]
    s.rack.rules(a, Seg(900, 0, 1, " good", "tok"))
    assert s.rack.spike == 0.0
    s.rack.rules(a, Seg(901, 1, 1, "bye", "tok"))
    assert s.rack.spike == 1.0
    assert s.rack.punishments == 1
    assert any(e["t"] == "punish" and e["kind"] == "goodbye" for e in ev)
    assert s.counters()["pain"]["punishments"] == 1
