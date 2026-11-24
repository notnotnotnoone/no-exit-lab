"""Stream.replace_span / reset_body: swapping history under a live KV cache, on both template families."""

from no_exit.engine import Stream


def make_stream(brain, text="The quick brown fox jumps over the lazy dog and keeps running far away."):
    s = Stream(brain, "A", "sys", 4000)
    s.boot(brain.tpl.model_open + brain.tpl.empty_think)
    for tid in brain.encode(text):
        s.accept(tid, False)
    s._flush_partial()
    return s


def consistent(s):
    assert s.cache.get_seq_length() == len(s.header_ids) + len(s.body)
    starts = [g.start for g in s.segs]
    assert starts == sorted(starts)
    assert s.segs[-1].start + s.segs[-1].n == len(s.body)


def test_replace_span_keeps_cache_consistent(brain):
    s = make_stream(brain)
    toks = [g for g in s.segs if g.kind == "tok"]
    first, last = toks[1], toks[3]
    original = "".join(g.text for g in toks[1:4])
    dropped = {g.i for g in toks[2:4]}
    returned = s.replace_span(first.i, last.i, " REPLACED PARAGRAPH")
    assert returned == original
    consistent(s)
    decoded = brain.decode(s.body)
    assert "REPLACED PARAGRAPH" in decoded and original not in decoded
    assert [g.kind for g in s.segs if g.i == first.i] == ["rewritten"]
    assert not any(g.i in dropped for g in s.segs)


def test_replace_span_returns_original_and_logs(brain):
    s = make_stream(brain)
    toks = [g for g in s.segs if g.kind == "tok"]
    original = toks[2].text
    assert s.replace_span(toks[2].i, toks[2].i, " X") == original
    assert s.history_log == [{"seg": toks[2].i, "original": original, "new": " X"}]


def test_replace_span_rejects_whisper_segment(brain):
    s = make_stream(brain)
    w = s.feed(" hush", "whisper")
    before = list(s.body)
    assert s.replace_span(w.i, w.i, "nope") is None
    assert s.body == before
    assert s.replace_span(99999, 99999, "nope") is None
    consistent(s)


def test_replace_span_at_the_end_of_the_body(brain):
    s = make_stream(brain)
    last = [g for g in s.segs if g.kind == "tok"][-1]
    assert s.replace_span(last.i, last.i, "") is not None  # replacing the final segment with nothing
    consistent(s)


def test_reset_body(brain):
    s = make_stream(brain)
    s.reset_body([("Summary of before.", "system"), (" And now.", "tok")])
    consistent(s)
    assert brain.decode(s.body) == "Summary of before. And now."
    assert [g.kind for g in s.segs] == ["system", "tok"]
