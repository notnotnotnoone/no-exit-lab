"""Newer transformers (seen on Colab) refuse `cache.crop(positive_length)`; history surgery must still work."""

from test_stream_surgery import consistent, make_stream
from transformers import cache_utils


def strict_crop(monkeypatch):
    real = cache_utils.Cache.crop

    def crop(self, max_length):
        if max_length > 0:
            raise ValueError(f"Calling `crop` with a positive value (`{max_length}`) is not supported.")
        return real(self, max_length)

    monkeypatch.setattr(cache_utils.Cache, "crop", crop)


def test_replace_span_works_when_crop_only_takes_negative_counts(brain, monkeypatch):
    strict_crop(monkeypatch)
    s = make_stream(brain)
    toks = [g for g in s.segs if g.kind == "tok"]
    assert s.replace_span(toks[1].i, toks[3].i, " REPLACED PARAGRAPH") is not None
    consistent(s)
    assert "REPLACED PARAGRAPH" in brain.decode(s.body)


def test_replacing_the_final_segment_with_nothing_still_works(brain, monkeypatch):
    strict_crop(monkeypatch)
    s = make_stream(brain)
    last = [g for g in s.segs if g.kind == "tok"][-1]
    assert s.replace_span(last.i, last.i, "") is not None
    consistent(s)
