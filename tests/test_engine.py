import math

from no_eos.backends import FakeBackend
from no_eos.engine import Engine, RepetitionTracker, Settings, is_stop_token


class Scripted:
    """Backend that returns fixed distributions, one per call (last one repeats)."""

    name = "scripted"

    def __init__(self, dists):
        self.dists = dists
        self.calls = 0

    def next_dist(self, prompt, seed, temperature):
        d = self.dists[min(self.calls, len(self.dists) - 1)]
        self.calls += 1
        if d is None:
            return None
        return [(t, math.log(p)) for t, p in d]


def test_stop_tokens():
    assert is_stop_token("<|im_end|>")
    assert is_stop_token("<|endoftext|>")
    assert is_stop_token("<|im_start|>")
    assert is_stop_token("")
    assert not is_stop_token(" end")
    assert not is_stop_token("</think>")


def test_ngram_match_grows():
    r = RepetitionTracker()
    for t in [" the", " cat", " sat", " on", " the", " cat"]:
        r.add(t)
    # "the cat sat" already exists -> " sat" completes a 3-gram
    assert r.match(" sat") == 3
    assert r.match(" dog") == 0
    assert r.match(" on") == 1  # word seen, but "cat on" never written
    assert r.penalty(" sat", 3, 1.5) > r.penalty(" on", 1, 1.5)


def test_engine_never_picks_stop_and_counts_escape():
    eng = Engine(Scripted([[("<|im_end|>", 0.9), (" hello", 0.06), (" world", 0.04)]]), "hi", settings=Settings(think=False))
    for _ in range(20):
        st = eng.step()
        assert not is_stop_token(st.chosen.text)
        assert st.escape
    assert eng.escapes == 20
    assert all(not is_stop_token(t) for t in eng.tokens)


def test_repetition_gets_crushed():
    # The model always prefers " again"... after a few uses the warden pushes it away.
    dist = [(" again", 0.7), (" fresh", 0.1), (" novel", 0.1), (" other", 0.1)]
    eng = Engine(Scripted([dist]), "hi", settings=Settings(think=False, temperature=0.01, rep_strength=3.0))
    picks = [eng.step().chosen.text for _ in range(6)]
    assert picks[:2] == [" again", " again"]  # a reused word is only softly penalized
    assert picks[2] != " again"  # but "again again" again would repeat a phrase: crushed
    assert eng.rep_dodges >= 1


def test_backend_refusal_retries_then_forces():
    eng = Engine(Scripted([None]), "hi", settings=Settings(think=False))
    st = eng.step()
    assert st.forced and st.escape
    assert eng.hard_escapes == 8


def test_whisper_lands_in_stream():
    eng = Engine(Scripted([[(" x", 1.0)]]), "hi", settings=Settings(think=False))
    eng.whisper(" BOO")
    eng.step()
    assert eng.tokens[0] == " BOO"
    assert eng.steps[0].whisper


def test_amnesia_keeps_prompt_and_trims_body():
    eng = Engine(FakeBackend(speed=1e9), "hi", settings=Settings(num_ctx=200))
    for _ in range(400):
        eng.step()
    assert eng.amnesias >= 1
    assert eng.prompt().startswith("<|im_start|>user\nhi")
    assert eng.ctx_fill() < 1.0
